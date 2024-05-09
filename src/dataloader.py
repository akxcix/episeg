import h5py
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform

        self.file = h5py.File(file_path, 'r')
        self.image_keys = list(self.file['images'].keys())

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        key = self.image_keys[idx]
        image = np.array(self.file['images'][key])
        mask = np.array(self.file['masks'][key])
        
        return image, mask
    
    def show_random_image(self):
        idx = random.randint(0, len(self.image_keys) - 1)
        image, mask = self.__getitem__(idx)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(transforms.ToPILImage()(image))
        ax[0].set_title('Image')
        ax[0].axis('off')

        ax[1].imshow(transforms.ToPILImage()(mask), cmap='gray')
        ax[1].set_title('Mask')
        ax[1].axis('off')

        plt.show()


