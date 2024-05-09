import os
import subprocess
import shutil
import logging

def prepare_datasets():
    DATA_LINK = "https://andrewjanowczyk.com/wp-static/epi.tgz"

    BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
    IMAGE_DIR = os.path.join(BASE_DIR, "data")
    MASK_DIR = os.path.join(IMAGE_DIR, "masks")
    TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train/images")
    TRAIN_MASK_DIR = os.path.join(IMAGE_DIR, "train/masks")
    TEST_DIR = os.path.join(IMAGE_DIR, "test")

    if not os.path.exists(IMAGE_DIR):
        logging.info("Downloading and extracting dataset...")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        print("Downloading and extracting dataset...")
        subprocess.run([
            "curl", "-L", DATA_LINK, "-o", os.path.join(IMAGE_DIR, "epi.tgz")
        ], check=True)
        subprocess.run(["tar", "-xzvf", os.path.join(IMAGE_DIR, "epi.tgz"), "-C", IMAGE_DIR], check=True)
        os.remove(os.path.join(IMAGE_DIR, "epi.tgz"))
        logging.info("Dataset downloaded and extracted.")

    os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
    os.makedirs(TRAIN_MASK_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    logging.info("Sorting images and masks...")
    for image in os.listdir(IMAGE_DIR):
        if image.endswith(".tif"):
            base = os.path.splitext(image)[0]
            mask_path = os.path.join(MASK_DIR, f"{base}_mask.png")

            if os.path.isfile(mask_path):
                shutil.move(os.path.join(IMAGE_DIR, image), TRAIN_IMAGE_DIR)
                shutil.move(mask_path, TRAIN_MASK_DIR)
            else:
                shutil.move(os.path.join(IMAGE_DIR, image), TEST_DIR)

    os.rmdir(MASK_DIR)

    logging.info("Images and masks sorted.")
