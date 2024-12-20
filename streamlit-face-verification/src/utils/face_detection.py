import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def detect_face(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Return True if at least one face is detected, otherwise False
    return len(faces) > 0

def calculate_average_distance(img_flat, imgs_flat, distance_metric):
    # Calc Average distance between first five images in imgs
    total_distance = 0
    for i in range(5):
        for j in range(i + 1, 5):
            if distance_metric == "euclidean":
                similarity = euclidean_distances([imgs_flat[i]], [imgs_flat[j]])
            elif distance_metric == "cosine":
                similarity = 1 - cosine_similarity([imgs_flat[i]], [imgs_flat[j]])  # 1-cosine for distance
            else:
                raise ValueError("Invalid distance metric. Choose 'euclidean' or 'cosine'.")
            total_distance += similarity[0][0]
    average_distance = total_distance / 10

    # Calc Average distance between new image and first five images in imgs
    total_distance = 0
    for i in range(5):
        if distance_metric == "euclidean":
            similarity = euclidean_distances([img_flat], [imgs_flat[i]])
        elif distance_metric == "cosine":
            similarity = 1 - cosine_similarity([img_flat], [imgs_flat[i]])  # 1-cosine for distance
        else:
            raise ValueError("Invalid distance metric. Choose 'euclidean' or 'cosine'.")
        total_distance += similarity[0][0]
    average_distance_with_new_image = total_distance / 5

    return average_distance, average_distance_with_new_image