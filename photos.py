import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import shutil
import urllib.request
from tqdm import tqdm
from datetime import datetime
import cv2

HOME = os.path.expanduser('~')
IMAGE_DIR = os.path.join(HOME, "media", "model")
SORTED_FOLDER = os.path.join(HOME, 'media', 'sorted')
# load the resnet model
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
model.eval()

# Define the image processing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load imagenet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


# Download the labels file
def download_labels(url):
    with urllib.request.urlopen(url) as f:
        return [line.decode("utf-8").strip() for line in f.readlines()]


def classify_image(image_path):
    """classify an image and return the top predicted label"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    return labels[predicted.item()], _


def organize():
    for name in image_names:
        image_path = os.path.join(IMAGE_DIR, name)
        assert os.path.exists(image_path), f'the path does not exist {image_path}'
        label = classify_image(image_path=image_path)
        print(label, name)
        

if __name__ == '__main__':
    # define the image directory

    assert os.path.exists(IMAGE_DIR), f'Path does not exist, {IMAGE_DIR}'
    assert os.path.exists(SORTED_FOLDER), f'Path does not exist, {SORTED_FOLDER}'

    # Download the labels
    labels = download_labels(LABELS_URL)
    image_names = os.listdir(IMAGE_DIR)

    organize()




