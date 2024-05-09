from data_preparation import prepare_datasets
import logging

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Preparing datasets...")
    prepare_datasets()
    logging.info("Datasets prepared.")

if __name__ == '__main__':
    main()