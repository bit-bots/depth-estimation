import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_loading import TorsoDataset

from tqdm import tqdm

from unet import UNet, FastDepth
from utils.transforms import combined_transforms

def evaluate(net, dataloader, device, progressbar=True, dataloader_len=None):
    net.eval()
    if dataloader_len is None:
        num_val_batches = len(dataloader)
    else:
        num_val_batches = dataloader_len
    val_score = 0.0
    rmse = 0.0
    rel = 0.0
    log10 = 0.0
    delta = [0.0, 0.0, 0.0]

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False, disable=not progressbar):
        image, mask_true = batch['image'], batch['depth']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32) / 255
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image).clamp(max=0.99, min=0.01)
            mask_pred = -1/(1-(1/mask_pred))
            
            # Use points with ground truth
            non_zero_points = (mask_true != 0).flatten()
            prep_pred = mask_pred.flatten()[non_zero_points]
            prep_true = mask_true.flatten()[non_zero_points]
            
            # Calc absolute metric
            err = torch.abs(prep_true-prep_pred)
            val_score += -err.mean().cpu() / num_val_batches
            # Calc root mean squared metric
            rmse += torch.pow(err, 2).mean().cpu() / num_val_batches
            # Calc relative metric (normalized by distance)
            rel += torch.mean(err / prep_true).cpu() / num_val_batches
            # Calc log10 metric
            log10 += torch.mean(torch.abs(torch.log10(prep_true) - torch.log10(prep_pred))).cpu() / num_val_batches
            # Calc delta 1-3 metrics
            delta_tmp = torch.max(torch.stack([prep_true/prep_pred, prep_pred/prep_true]), axis=0)[0]
            for j in range(len(delta)):
                delta[j] += (delta_tmp < (1.25**(j+1))).float().mean() / num_val_batches

    # Root for root mean squared error
    rmse = torch.sqrt(rmse)

    net.train()
    return val_score, rmse, rel, log10, delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the network on a given dataset')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='4', type=int, default=32, help='Batch size')
    parser.add_argument('--load', '-f', type=str, help='Load model from a .pth file')
    parser.add_argument('--data-path', type=str, help='Path to the test data')
    parser.add_argument('--model', type=str, help='Name of the models class')

    args =  parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model.lower() == "unet":
        model = UNet(n_channels=3, bilinear=True)
    elif args.model.lower() == "fastdepth":
        model = FastDepth()
    else:
        print("Unknown model type")

    model.load_state_dict(torch.load(args.load, map_location=device))
    print(f'Model loaded from {args.load}')
   
    model.to(device)

    dataset = TorsoDataset(Path(args.data_path) / Path('images'), Path(args.data_path) / Path('depth'), transform=combined_transforms)

    dataloader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=24, pin_memory=True)

    result = evaluate(model, dataloader, device, progressbar=True, dataloader_len=len(dataloader))

    print(f"Abs: {result[0]} | RMSE: {result[1]} | Relative Error: {result[2]} | Log10: {result[3]} | Delta (1-3): {' '.join([str(float(element)) for element in result[4]])}")

