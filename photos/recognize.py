import face_recognition
import os
import numpy as np

# Define known people
HOME = os.path.expanduser('~')



# Define directories
KNOWN_PEOPLE_DIR = os.path.join(HOME, 'media', 'model', 'known_faces')
KNOWN_DOGS_DIR = os.path.join(HOME, 'media', 'model', 'known_dogs')
assert os.path.exists(KNOWN_PEOPLE_DIR), f'the path does not exist {KNOWN_PEOPLE_DIR}'
assert os.path.exists(KNOWN_DOGS_DIR), f'the path does not exist {KNOWN_DOGS_DIR}'

# known_faces = []
# known_names = []


def load_known_entities(directory, label_prefix):
    """Loads known faces or dogs and their encodings"""
    known_entities = []
    known_labels = []

    for label in os.listdir(directory):
        person_or_dog_dir = os.path.join(directory, label)
        print(f'loading from {person_or_dog_dir}')
        if os.path.isdir(person_or_dog_dir):
            for image_name in os.listdir(person_or_dog_dir):
                image_path = os.path.join(person_or_dog_dir, image_name)

                # Skip non-image files
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    continue

                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_entities.append(encodings[0])
                        known_labels.append(f"{label_prefix}{label}")
                except Exception as e:
                    print(f"Skipping {image_path}: {e}")

    return known_entities, known_labels


def recognize_entity(image_path):
    """Recognizes a human or dog in an image"""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        if True in matches:
            matched_idx = np.argmin(face_recognition.face_distance(known_faces, face_encoding))
            return known_names[matched_idx]
    return "unknown"


# Load known people and dogs
known_faces, known_names = load_known_entities(KNOWN_PEOPLE_DIR, "human_")
known_dog_faces, known_dog_names = load_known_entities(KNOWN_DOGS_DIR, "dog_")

# Merge both lists
known_faces.extend(known_dog_faces)
known_names.extend(known_dog_names)