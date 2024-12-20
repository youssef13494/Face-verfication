import cv2
import numpy as np
import streamlit as st
from utils.face_detection import detect_face, calculate_average_distance
from PIL import Image

def detect_faces_and_draw_boxes(frame, gray_frame, imgs_flat, distance_metric):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (960, 1280))
        face_flat = face_resized.flatten()

        # Calculate average distance
        average_distance, average_distance_with_new_image = calculate_average_distance(face_flat, imgs_flat, distance_metric)

        # Determine the threshold based on the chosen metric
        if distance_metric == "euclidean":
            threshold = 35000
        elif distance_metric == "cosine":
            threshold = 0.1  # Adjust as needed

        # Check if face matches
        if abs(average_distance_with_new_image - average_distance) < threshold:
            color = (0, 255, 0)  # Green for True
            label = "Face: True"
        else:
            color = (0, 0, 255)  # Red for False
            label = "Face: False"

        # Draw a rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return faces


def main():
    st.title("Live Video Face Verification")

    # Choose the distance metric (cosine or euclidean)
    distance_metric = st.selectbox("Choose distance metric", ["euclidean", "cosine"])

    # Load reference images (Youssef1.jpg to Youssef5.jpg + Komy.jpg)
    imgs = []
    for i in range(5):
        img = cv2.imread(f'Youssef{i+1}.jpg', 0)
        img = cv2.resize(img, (960, 1280))
        imgs.append(img)

    # Flatten reference images
    imgs_flat = [img.flatten() for img in imgs]

    # Start video capture
    cap = cv2.VideoCapture(0)  # 0 for default camera

    if not cap.isOpened():
        st.error("Error: Could not access the camera.")
        return

    st.text("Starting live video...")

    # Streamlit video display container
    video_placeholder = st.empty()

    # Stop video button
    stop = st.button("Stop Video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from camera.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detect_faces_and_draw_boxes(frame, gray_frame, imgs_flat, distance_metric)

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)

        # Display the video frame in Streamlit
        video_placeholder.image(frame_image, use_container_width=True)

        # Stop the video if button is pressed
        if stop:
            st.write("Video Stopped.")
            break

    cap.release()



if __name__ == "__main__":
    main()
