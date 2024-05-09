import data
import dataset
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import models
import train

DATA_LINK = "https://andrewjanowczyk.com/wp-static/epi.tgz"
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
IMAGE_DIR = os.path.join(BASE_DIR, "data")
MASK_DIR = os.path.join(IMAGE_DIR, "masks")
TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(IMAGE_DIR, "train/masks")
TEST_DIR = os.path.join(IMAGE_DIR, "test")
HDF5_FILE_PATH = os.path.join(IMAGE_DIR, "train_data.h5")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_checkpoint.pth")

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Preparing datasets...")
    data.prepare_datasets(DATA_LINK, IMAGE_DIR, MASK_DIR, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TEST_DIR)

    logging.info("Loading dataset...")
    dataset = dataset.SegmentationDataset(HDF5_FILE_PATH)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    logging.info("Setting up the model...")
    model = models.UNet(n_channels=3, n_classes=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train.train_model(model, data_loader, 10, optimizer, criterion, device, CHECKPOINT_PATH)

    dataset.show_random_image()

if __name__ == '__main__':
    main()
