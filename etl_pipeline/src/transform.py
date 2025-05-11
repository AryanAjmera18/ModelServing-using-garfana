from torchvision import transforms
from PIL import Image
import torch
import os

def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def transform_images(raw_dir, processed_dir, image_size, class_map, logger):
    os.makedirs(processed_dir, exist_ok=True)
    transform = get_transform(image_size)
    logger.info("Starting image transformation...")

    total_transformed = 0
    total_skipped = 0

    for class_name in os.listdir(raw_dir):
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        label = class_map.get(class_name, -1)
        if label == -1:
            logger.warning(f"Skipping unknown class: {class_name}")
            continue

        target_class_path = os.path.join(processed_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # Skip non-image files

            img_path = os.path.join(class_path, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                transformed_img = transform(img)

                # Save .pt tensor
                tensor_path = os.path.join(target_class_path, img_file.replace(".jpg", ".pt"))
                torch.save({"tensor": transformed_img, "label": label}, tensor_path)

                # Also save resized .jpg
                resized_img = transforms.ToPILImage()(transformed_img)
                img_save_path = os.path.join(target_class_path, img_file)
                resized_img.save(img_save_path)

                total_transformed += 1
            except  Exception as e:
                logger.warning(f"[Transform] Skipping {img_file}: {e}")
                total_skipped += 1

    logger.info(f"Total images transformed: {total_transformed}")
    logger.info(f"Total images skipped: {total_skipped}")