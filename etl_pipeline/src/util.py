import yaml  
import json
import logging

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_class_map(path):
    with open(path, 'r') as file:
        return json.load(file)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()