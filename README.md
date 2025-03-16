# Live Video Stream Object Detection Using OpenCV and PyTorch

This project demonstrates real-time object detection using a deep learning-based approach with **PyTorch** and the **Faster R-CNN** model. The application leverages a pre-trained model for accurate object detection, including bounding boxes and prediction confidence scores. The detection is performed using a webcam feed, and detected objects are labeled with their class names and confidence scores.

## Table of Contents

1. [Overview](#overview)
2. [Theory](#theory)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Details](#model-details)
6. [File Structure](#file-structure)
7. [Contributing](#contributing)
8. [License](#license)

## Overview

This project uses a pre-trained **Faster R-CNN** model from **TorchVision** to perform real-time object detection via webcam. The model not only detects objects but also provides a confidence score for each detected object. The webcam feed is processed, and bounding boxes along with object labels and confidence percentages are displayed on the frame.

The application is developed using **OpenCV** for handling webcam input and **PyTorch** for the deep learning model. The results are shown in real-time as a video stream.

## Theory

### Object Detection

Object detection is the computer vision task of identifying and classifying multiple objects within images or video frames. It is used in various applications, such as autonomous driving, security surveillance, and smart retail systems. In this project, we use a deep learning-based object detection method, specifically the **Faster R-CNN** (Region-based Convolutional Neural Networks).

### Faster R-CNN

**Faster R-CNN** is a deep learning model used for object detection. It is one of the most accurate models available and works by:

1. **Region Proposal Network (RPN):** It generates potential bounding boxes for objects in the image. The RPN generates a set of candidate regions, also known as region proposals.
   
2. **ROI Pooling:** This step takes the candidate regions proposed by the RPN and extracts fixed-size feature maps.
   
3. **Classifier and Bounding Box Regressor:** The model uses these feature maps to classify objects and adjust the bounding boxes to better fit the objects.

The model outputs:
- **Bounding boxes:** The coordinates of the detected object.
- **Labels:** The class of the detected object (e.g., person, car, dog, etc.).
- **Scores:** Confidence levels that indicate how certain the model is about the prediction.

### Confidence Score

The **confidence score** is a value between 0 and 1 that indicates how certain the model is about the prediction. A higher score means the model is more confident about its detection.

In this project, the object detection model assigns a confidence score to each detected object. If the confidence score exceeds a predefined threshold (e.g., 50%), the detection is considered valid, and the bounding box is displayed on the webcam feed along with the class name and confidence percentage.

## Installation

To set up and run this project, follow these steps:

### Prerequisites

1. **Python 3.6+**  
2. **pip** (Python package installer)

### Step-by-Step Setup

1. **Clone the repository:**
   
   ```bash
   git clone https://github.com/bastinjob/object-detection.git
   cd object-detection
   ```

2. Create a virtual environment (optional but recommended):
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use venv\Scripts\activate
    ```

3. Install the required dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
    - `torch` and `torchvision`: Required for running the Faster R-CNN model.
    - `opencv-python`: For handling webcam input and displaying frames.
    - `matplotlib`: For image visualization (optional).

4. Verify the installation:

    Ensure that PyTorch and OpenCV are installed correctly by running the following commands in Python:
    
    ```python
    import torch
    import cv2
    print(torch.__version__)
    print(cv2.__version__)
    ```

## Usage

1. Run the application:

    To start the real-time object detection, run the following command in the terminal:
    
    ```bash
    python object_detection.py
    ```

2. Operation:
    
    - The program will open your webcam feed and start processing the video stream.
    - Detected objects will have bounding boxes drawn around them, along with labels and confidence percentages.
    - To exit the application, press the `q` key.

3. Adjusting Confidence Threshold:
    
    You can adjust the confidence threshold in the code to control which detections are shown. The default threshold is set to 50%, but you can modify it in the script:
    
    ```python
    if score > 0.5:  # Modify this value to adjust the threshold
    ```

## Model Details
### Faster R-CNN Model

The Faster R-CNN model used in this project is based on ResNet-50 architecture and uses Feature Pyramid Networks (FPN) for improved object detection performance. It is a state-of-the-art model for detecting various objects and is available through TorchVision.

- **Pretrained on COCO dataset**: The model has been pre-trained on the COCO (Common Objects in Context) dataset, which contains a wide variety of images with annotated objects.
- **Output**: The model outputs bounding boxes, labels, and confidence scores for each detected object.

For more details on Faster R-CNN, refer to the official PyTorch documentation.

## File Structure

```bash
object-detection/
│
├── object_detection.py       # Main script for object detection
├── requirements.txt          # List of dependencies
└── README.md                 # This documentation
```

- `object_detection.py`: This is the main Python script that runs the object detection.
- `requirements.txt`: This file lists all the necessary Python packages to run the project.
- `README.md`: The documentation you're currently reading.

## Deployment Strategies
1. Containerization with Docker

Containerization allows you to encapsulate the entire application, including its dependencies, into a portable container that can be run consistently across different environments. Using Docker to containerize the application provides several benefits:

- Consistency: The application will run the same way across all systems, regardless of the underlying infrastructure.
- Portability: Docker containers can be deployed on any system that supports Docker, including local machines, virtual machines, and cloud platforms.
- Isolation: Docker ensures that the application runs in its own environment, preventing conflicts with other applications running on the same host system.

To containerize the application, we use Docker to create a Dockerfile, which defines the steps to set up the environment, install dependencies, and run the application. Once the container image is built, it can be deployed to any system, including cloud platforms, for scalable and isolated execution.

2. Web API Deployment

By wrapping the face detection logic into a REST API, you can deploy the application to be accessed over the web. This allows you to serve real-time video streams with face detection capabilities to remote clients.

- Web Access: The application can be accessed from any device with a browser, making it easier to integrate into web or mobile applications.
- Scalability: The API can be deployed on cloud platforms to scale based on the number of incoming requests.
- User Interaction: Clients can interact with the API through simple HTTP requests, providing flexibility for integrating the model into various systems.

To deploy as a web service, the Flask or FastAPI framework is used to create endpoints that stream the webcam feed with face detection. Once the API is set up, it can be deployed to a server or cloud platform (e.g., AWS, Google Cloud) for production usage.

## Contributing

Contributions to this project are welcome! To contribute:

- Fork the repository.
- Create a new branch (`git checkout -b feature-name`).
- Make your changes and commit them (`git commit -am 'Add new feature'`).
- Push the branch (`git push origin feature-name`).
- Create a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.