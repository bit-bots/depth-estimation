import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


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
