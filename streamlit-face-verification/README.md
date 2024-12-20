# Streamlit Face Verification

This project implements a video face verification application using Streamlit. The application detects faces in real-time video input and displays "True" when a face is detected and "False" when no face is detected.

## Project Structure

```
streamlit-face-verification
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   └── utils
│       └── face_detection.py  # Utility functions for face detection
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-face-verification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application, execute the following command:
```
streamlit run src/app.py
```

Open your web browser and navigate to `http://localhost:8501` to access the application.

## Dependencies

This project requires the following Python packages:
- Streamlit
- OpenCV
- face_recognition

Make sure to install these packages using the `requirements.txt` file provided.