import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from kornia.filters import spatial_gradient, sobel

from utils.data_loading import BasicDataset, TorsoDataset #  CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet, FastDepth

dir_checkpoint = Path('./checkpoints/')

WANDB = True

def train_net(net,
              device,
              trainpath,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              save_checkpoint: bool = True,
              amp: bool = False,
              weight_decay=1e-12,
              momentum=0.9,
              sched_gamma=0.16,
              progressbar=True,
              validation_split=1/6,
              loss_weighting=0.5,
              sparse_labels=False,
              trial=None):
    # 1. Create dataset

    #made new dataset, comment is the old version
    dataset = TorsoDataset(Path(trainpath) / Path('images'), Path(trainpath) / Path('depth'))

    # 2. Split into train / validation partitions
    n_val = int(round(len(dataset) * validation_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=16, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    num_val_batches = len(val_loader)

    # (Initialize logging when not using optuna)
    if trial is None:
        #experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
        experiment = wandb.init(project='depth-u-net-paper', resume='allow', entity="bitbots")
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                      save_checkpoint=save_checkpoint, amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=sched_gamma)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=7)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', disable=not progressbar) as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'


                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32) #dtype=torch.long
                true_masks = 1-1/(true_masks+1) #Transform values to a 0 to 1 value
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)

                    # Check if we have spase labels and only want to backprop non zero values with an MSE loss
                    if sparse_labels:
                        loss = (torch.pow(true_masks - masks_pred, 2) * true_masks.bool().int().float().clamp(min=0.001, max=1.0)).sum() / true_masks.bool().sum()
                    else:
                        distance_loss = torch.pow(true_masks - masks_pred, 2).mean()
                        d_true = sobel(true_masks)
                        d_pred = sobel(masks_pred)
                        img_grad_loss = torch.mean(torch.abs(d_pred - d_true))
                        loss = loss_weighting * distance_loss + (1 - loss_weighting) * img_grad_loss
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                if trial is None:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })

                # Evaluation round
                division_step = n_train // (batch_size * 4)
                if global_step  % division_step == 0:

                    val_score = evaluate(net, val_loader, device, progressbar=progressbar, dataloader_len=num_val_batches)
                    scheduler.step()
                    if trial:
                        trial.report(val_score, (global_step // division_step)-1)
                        # Handle pruning based on the intermediate value.
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                    else:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        logging.info('Validation score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

    return val_score

def run_trial(trial, args):

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-12, 1e-6)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    bilinear = trial.suggest_categorical('bilinear', [True, False])
    sched_gamma = trial.suggest_float('sched_gamma', 0, 0.2, step=0.02)
    loss_weighting = trial.suggest_float('loss_weighting', 0, 1, step=0.1)
    momentum = trial.suggest_float('momentum', 0.8, 0.99, step=0.01)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Change here to adapt to your data
    # n_channels=3 for RGB images

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    net = UNet(n_channels=3, bilinear=bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    return train_net(
        net=net,
        trainpath=args.train_path,
        epochs=args.epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        amp=args.amp,
        save_checkpoint=False,
        sched_gamma=sched_gamma,
        loss_weighting=loss_weighting,
        weight_decay=weight_decay,
        momentum=momentum,
        progressbar=False,
        sparse_labels=args.spare_labels,
        trial=trial)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='4', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--gpu', type=str, default='0', help='Specify gpu device')
    parser.add_argument('--optuna', '-o', action='store_true', default=False, help='Use optuna optimization')
    parser.add_argument('--study-name', type=str, default='depth-unet-0', help='Name of the optuna study')
    parser.add_argument('--study-storage', type=str, default='', help='Storage of the optuna study')
    parser.add_argument('--train-path', type=str, help='Path to the training data')
    parser.add_argument('--sparse-labels', '-sl', action='store_true', default=False, \
        help='Enables training with sparse labels that contain no information for certain regions. \
        The unknown regions are defined by the 0.0 value.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.optuna:
        import optuna
        logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

        if args.study_storage == '':
            study = optuna.create_study(direction='maximize')
        else:
            study = optuna.load_study(study_name=args.study_name, storage=args.study_storage)
        study.optimize(lambda t: run_trial(t, args), timeout=24*60*60)
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        #net = FastDepth()
        net = UNet(n_channels=3, bilinear=True)

        logging.info(f'Network:\n'
                     f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

        if args.load:
            net.load_state_dict(torch.load(args.load, map_location=device))
            logging.info(f'Model loaded from {args.load}')

        net.to(device=device)
        try:
            train_net(net=net,
                      trainpath=args.train_path,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      learning_rate=args.lr,
                      device=device,
                      amp=args.amp,
                      sparse_labels=args.sparse_labels)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            sys.exit(0)

