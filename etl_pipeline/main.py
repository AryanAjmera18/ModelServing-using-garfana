from src.extract import extract_raw_data
from src.transform import transform_images
from src.util import load_config,load_class_map,setup_logger
import os
import logging

CONFIG_PATH = os.path.join("config", "config.yaml")
CLASS_MAP_PATH = "class_map.json"

if __name__ == "__main__":
    logger = setup_logger()
    config = load_config(CONFIG_PATH)
    class_map = load_class_map(CLASS_MAP_PATH)

    logger.info("🚀 Starting ETL Pipeline")
    extract_raw_data(r"D:\MLopsProject\etl_pipeline\data\raw\Original_Dataset", config['raw_data_dir'], logger)



    transform_images(config['raw_data_dir'], config['processed_data_dir'], config['image_size'], class_map, logger)
    logger.info("✅ ETL Pipeline Completed")
