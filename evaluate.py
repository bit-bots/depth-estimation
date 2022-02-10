import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long) 

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = 1/(1-1/(mask_pred+1)) 
            err = torch.abs(mask_true-mask_pred)
            a,b,c,d = err.size()
            dice_score = torch.sum(err).cpu()/(a*b*c*d)
            
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
