import os
import shutil
from pathlib import Path
import logging

def extract_raw_data(source_dir, target_dir, logger):
    os.makedirs(target_dir, exist_ok=True)
    logger.info("Starting raw data extraction...")
    
    total_files = 0
    for folder in os.listdir(source_dir):
        full_path = os.path.join(source_dir, folder)
        if os.path.isdir(full_path):
            dest_path = os.path.join(target_dir, folder)
            shutil.copytree(full_path, dest_path, dirs_exist_ok=True)
            count = len(os.listdir(dest_path))
            total_files += count
            logger.info(f"✔️ Copied {folder} with {count} files.")

    logger.info(f"Total files extracted: {total_files}")
