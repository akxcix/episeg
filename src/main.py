import data
import dataloader
import logging
import os

DATA_LINK = "https://andrewjanowczyk.com/wp-static/epi.tgz"

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
IMAGE_DIR = os.path.join(BASE_DIR, "data")
MASK_DIR = os.path.join(IMAGE_DIR, "masks")
TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(IMAGE_DIR, "train/masks")
TEST_DIR = os.path.join(IMAGE_DIR, "test")
HDF5_FILE_PATH = os.path.join(IMAGE_DIR, "train_data.h5")

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Preparing datasets...")
    data.prepare_datasets(DATA_LINK, IMAGE_DIR, MASK_DIR, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TEST_DIR)
    data.create_h5_files(HDF5_FILE_PATH, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)

    dataset = dataloader.SegmentationDataset(HDF5_FILE_PATH)
    dataset.show_random_image()


    

if __name__ == '__main__':
    print(os.path.dirname(__file__))
    main()