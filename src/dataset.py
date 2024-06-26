import h5py
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import logging

class RandomTransforms:
    def __init__(self):
        self.random_crop = transforms.RandomCrop(256)

    def __call__(self, image, mask):
        i, j, h, w = self.random_crop.get_params(image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        angle = random.uniform(-30, 30)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        return image, mask

class SegmentationDataset(Dataset):
    def __init__(self, file_path):
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(64, padding_mode='reflect')
        ])
        self.to_tensor = transforms.ToTensor()
        self.random_transforms = RandomTransforms()

        self.file = h5py.File(file_path, 'r')
        self.image_keys = list(self.file['images'].keys())

    def __len__(self):
        return len(self.image_keys)
    
    def __getitem__(self, idx):
        key = self.image_keys[idx]
        image = np.array(self.file['images'][key])
        mask = np.array(self.file['masks'][key])

        image = self.base_transform(image)
        mask = self.base_transform(mask)

        image, mask = self.random_transforms(image, mask)

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        return image, mask
    
    def plot_image(self, image, mask):
        image_display = transforms.ToPILImage()(image)
        mask_display = transforms.ToPILImage()(mask)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image_display)
        ax[0].set_title('Image')
        ax[0].axis('off')
        ax[1].imshow(mask_display, cmap='gray')
        ax[1].set_title('Mask')
        ax[1].axis('off')
        plt.show()

    def plot_with_pred(self, image, mask, pred, thresholds=[0.5]):
        image_display = transforms.ToPILImage()(image)
        mask_display = transforms.ToPILImage()(mask)

        fig, ax = plt.subplots(1, 2+len(thresholds), figsize=(18, 6))
        ax[0].imshow(image_display)
        ax[0].set_title('Image')
        ax[0].axis('off')
        ax[1].imshow(mask_display, cmap='gray')
        ax[1].set_title('Mask')
        ax[1].axis('off')
        for idx in range(len(thresholds)):
            ax_idx = 2+idx
            threshold = thresholds[idx]
            masked_pred = (pred>threshold).astype(np.uint8)
            pred_display = transforms.ToPILImage()(masked_pred)
            ax[ax_idx].imshow(pred, cmap='gray')
            ax[ax_idx].imshow(pred_display, cmap='gray')
            ax[ax_idx].set_title(f'Prediction: {threshold}')
            ax[ax_idx].axis('off')
        plt.show()
    
    def show_image(self, idx=None):
        if idx is None:
            idx = random.randint(0, len(self.image_keys) - 1)

        logging.debug(f"showing image at idx: {idx}")
        image, mask = self.__getitem__(idx)
        self.plot_image(image, mask)
