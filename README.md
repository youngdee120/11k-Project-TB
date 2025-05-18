# README.md

# TB-Scan Web App

A simple Flask application for detecting tuberculosis (TB) from chest X-ray images using a pre-trained Keras model.

## Features

- Drag-and-drop or browse to upload an X-ray image (also supports paste from clipboard)  
- Live preview of the selected image  
- Server-side inference with your `tbx11k_classifier.h5` model  
- Results page showing:
  - The uploaded X-ray  
  - A label (“TB Present” or “TB absent”)  
  - Confidence scores for both classes (bar chart)  

## Project Structure

11K PROJECT TB/
├── app.py
├── static/
│ ├── css/
│ │ └── styles.css
│ ├── models/
│ │ └── tbx11k_classifier.h5
│ └── uploads/ ← (created at runtime)
├── templates/
│ ├── index.html
│ └── results.html
├── requirements.txt
└── README.md



## Setup & Installation

1. **Clone the repo**  
   git clone <your-repo-url>
   cd your_project

# Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate.bat     # Windows

# Install dependencies

pip install -r requirements.txt


# Running the App

python app.py
