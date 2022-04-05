
import cv2

from albumentations.pytorch import ToTensorV2
import albumentations as A


geometric_transforms = A.Compose(
    [
        # A.ShiftScaleRotate(shift_limit=0.5, scale_limit=[0.0, 0.2], rotate_limit=[-20, 20], border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip()
    ],
    additional_targets={'depth': 'image', 'image': 'image'}
)

rgb_transforms = A.Compose(
    [   
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3)
    ],
    additional_targets={'image': 'image'}
)

combined_transforms = A.Compose(
    [   
        rgb_transforms,
        geometric_transforms,
	A.Compose([ 
            A.Resize(128, 128, interpolation=cv2.INTER_AREA)
        ], additional_targets={'image': 'image'}),
        A.Compose([
            A.Resize(128, 128, interpolation=cv2.INTER_NEAREST)
        ], additional_targets={'depth': 'image'}),
        A.Compose([
            ToTensorV2()
        ], additional_targets={'depth': 'image', 'image': 'image'})
    ]
)
