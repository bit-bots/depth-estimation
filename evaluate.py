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
    val_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False, disable=not progressbar):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image).clamp(max=0.99, min=0.01)
            mask_pred = -1/(1-(1/mask_pred))
            err = torch.abs(mask_true-mask_pred)
            val_score += -torch.mean(err).cpu() / num_val_batches
    net.train()
    return val_score
