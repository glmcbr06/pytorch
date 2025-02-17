import face_recognition
import os
import numpy as np

# Define known people
HOME = os.path.expanduser('~')
KNOWN_FACES_DIR = os.path.join(HOME, 'media', 'model', 'known_faces')
assert os.path.exists(KNOWN_FACES_DIR), f'the path does not exist {KNOWN_FACES_DIR}'
known_faces = []
known_names = []


def load_known_faces():
    """Loads known faces and their encodings"""
    global known_faces, known_names
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        print(f'loading images for {person_dir}')
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_faces.append(encodings[0])
                    known_names.append(person_name)


def recognize_face(image_path):
    """Detects and recognizes a face in an image"""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        if True in matches:
            matched_idx = np.argmin(face_recognition.face_distance(known_faces, face_encoding))
            return known_names[matched_idx]
    return "unknown"

# Load known faces at script start
load_known_faces()

