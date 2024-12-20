# Live Video Face Verification

This project implements live video face verification using a webcam. It uses OpenCV for face detection and calculates distances between a new image and a set of reference images to determine if a face matches a pre-stored reference.

## Features

- Real-time face detection and verification using a webcam.
- Two distance metrics to calculate face similarity: Euclidean and Cosine.
- Displays green or red rectangle around the face indicating whether the face matches any of the reference images.

## Requirements

The following Python libraries are required to run this project:

- `opencv-python`
- `numpy`
- `streamlit`
- `Pillow`
- `scikit-learn`

You can install them by running the following command:

```bash
pip install -r requirements.txt
