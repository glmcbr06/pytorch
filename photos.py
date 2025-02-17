import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from PIL.ExifTags import TAGS
import os
import shutil
import urllib.request
from tqdm import tqdm
from datetime import datetime
import cv2


HOME = os.path.expanduser('~')
IMAGE_DIR = os.path.join(HOME, "media", "model")
SORTED_FOLDER = os.path.join(HOME, 'media', 'sorted')
# Define a mapping for high-level categories
CATEGORY_MAPPING = {
    "dog": ["german shepherd", "labrador retriever", "golden retriever", "bulldog", "beagle", "poodle", "chihuahua"],
    "cat": ["siamese cat", "persian cat", "maine coon", "sphinx", "tabby"],
    "bird": ["parrot", "eagle", "sparrow", "penguin", "owl"],
    "vehicle": ["car", "truck", "motorcycle", "bus", "bicycle"],
    "person": ["man", "woman", "child", "boy", "girl"]
}

haar_cascade_path = os.path.join(cv2.__path__[0], "data", "haarcascade_frontalface_default.xml")

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


def map_to_high_level_category(label):
    """Map a specific label to a high-level category."""
    for category, sub_labels in CATEGORY_MAPPING.items():
        if any(sub_label in label.lower() for sub_label in sub_labels):
            return category
    return label  # Default to the original label if no match is found


def get_photo_date(image_path):
    """Extracts the creation date of a photo from metadata."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d")
    except Exception as e:
        print(f"EXIF read error for {image_path}: {e}")

    return "unknown_date"


def classify_image(image_path):
    """classify an image and return the top predicted label"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    return map_to_high_level_category(labels[predicted.item()])


def detect_faces(image_path):
    """Detect faces in an image and return whether a person is present."""
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0


def organize():
    for name in image_names:
        image_path = os.path.join(IMAGE_DIR, name)
        has_face = detect_faces(image_path)
        face_label = "_person" if has_face else ""
        date = get_photo_date(image_path)
        assert os.path.exists(image_path), f'the path does not exist {image_path}'
        label = classify_image(image_path=image_path)
        print(label, face_label, date, name)


if __name__ == '__main__':
    # define the image directory

    assert os.path.exists(IMAGE_DIR), f'Path does not exist, {IMAGE_DIR}'
    assert os.path.exists(SORTED_FOLDER), f'Path does not exist, {SORTED_FOLDER}'

    # Download the labels
    labels = download_labels(LABELS_URL)
    image_names = os.listdir(IMAGE_DIR)

    organize()




