--- 
title: Real-Time Face Detection and Sketch Generation
date: 2020-11-11
categories: [Projects, Image Processing] 
tags: [Machine Learning, Face Recognition, Open CV, Image Processing, Python]

---

[![Open in Github Page](https://img.shields.io/badge/Hosted_with-GitHub_Pages-blue?logo=github&logoColor=white)](https://github.com/AbhijitMore/FaceDetectionApp)
<br>


# Real-Time Face Detection and Sketch Generation

This is a simple Flask application that streams video from a webcam while performing facial recognition using OpenCV's Haar Cascade Classifier. The application captures video frames, detects faces, and displays them on a web page in real-time.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

## Getting Started

To get started with this facial recognition video streaming app, follow the instructions below.

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Flask
- OpenCV

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/FaceDetectionApp.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd FaceDetectionApp
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Flask application:**

    ```bash
    python app.py
    ```

2. **Open your web browser and go to [http://localhost:5000/](http://localhost:5000/) to access the facial recognition video streaming page.**

3. **Navigate to the `/video_feed` route to view the live video stream with facial recognition.**

## Customization

If you want to customize the application or integrate it with a different camera source, you can modify the following files:

- `app.py`: Main application file containing the Flask routes.
- `camera.py`: Module defining the `VideoCamera` class responsible for capturing video frames and performing facial recognition.

Feel free to explore and adapt the code to meet your specific facial recognition requirements.
Feel free to reach out if you have any questions or need further assistance with this facial recognition video streaming app!
