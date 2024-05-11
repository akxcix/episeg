import os
import subprocess
import shutil
import logging
import h5py
import numpy as np
from PIL import Image

def prepare_datasets(data_link, image_dir, mask_dir, train_image_dir, train_mask_dir, test_dir):
    if not os.path.exists(image_dir):
        logging.info("downloading dataset...")
        os.makedirs(image_dir, exist_ok=True)
        subprocess.run([
            "curl", "-k", "-L", data_link, "-o", os.path.join(image_dir, "epi.tgz")
        ], check=True)
        logging.info("dataset downloaded.")
        logging.info("extracting dataset...")
        subprocess.run(["tar", "-xzvf", os.path.join(image_dir, "epi.tgz"), "-C", image_dir], check=True)
        os.remove(os.path.join(image_dir, "epi.tgz"))
        logging.info("dataset extracted.")

    logging.info("creating directories...")
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(data_link, exist_ok=True)
    logging.info("directories created.")

    logging.info("sorting images and masks...")
    for image in os.listdir(image_dir):
        if image.endswith(".tif"):
            base = os.path.splitext(image)[0]
            mask_path = os.path.join(mask_dir, f"{base}_mask.png")

            if os.path.isfile(mask_path):
                shutil.move(os.path.join(image_dir, image), train_image_dir)
                shutil.move(mask_path, train_mask_dir)
            else:
                shutil.move(os.path.join(image_dir, image), test_dir)

    logging.info("images and masks sorted.")

def create_h5_files(hdf5_file_path, train_image_dir, train_mask_dir):
    logging.info("creating hdf5 file with images and masks...")
    with h5py.File(hdf5_file_path, 'w') as h5f:
        logging.info("hdf5 file created.")
        logging.info("creating datasets...")
        image_dataset = h5f.create_group('images')
        mask_dataset = h5f.create_group('masks')
        logging.info("datasets created.")

        for image_name in os.listdir(train_image_dir):
            if image_name.endswith(".tif"):
                image_path = os.path.join(train_image_dir, image_name)
                base = os.path.splitext(image_name)[0]
                mask_path = os.path.join(train_mask_dir, f"{base}_mask.png")
                
                image = np.array(Image.open(image_path))
                mask = np.array(Image.open(mask_path))
                
                image_dataset.create_dataset(base, data=image, compression="gzip")
                mask_dataset.create_dataset(base, data=mask, compression="gzip")
        
    logging.info("hdf5 file created with images and masks.")
