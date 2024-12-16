# Emotion-Detection-from-Uploaded-Images

## Objective
This project aims to develop a comprehensive system that detects and classifies emotions from uploaded images using Convolutional Neural Networks (CNNs). The system is designed to be user-friendly, efficient, and robust, combining machine learning, computer vision, and UI development.

---

## Features
1. **Image Upload via Streamlit**:  
   - A responsive and intuitive web interface for users to upload images.
   - Restricts uploads to image files only (validated by format and size checks).

2. **Facial Detection**:  
   - Implements facial detection using pre-trained models and custom-built models.  
   - Focuses on optimizing precision, recall, and F1 score by refining detection thresholds.

3. **Facial Feature Extraction**:  
   - Extracts key facial landmarks using tools like **Dlib** or **Mediapipe**.  
   - Analyzes the impact of landmark detection accuracy on emotion classification.

4. **Emotion Classification**:  
   - Trains and fine-tunes CNN models using the **FER-2013 dataset**.  
   - Explores and compares the performance of three CNN architectures:
     - **MobileNetV2**
     - **ResNet50**
     - **InceptionV3**

5. **Performance Optimization**:  
   - Evaluates model performance with metrics such as accuracy, precision, recall, and F1 score.  
   - Ensures efficient, real-time emotion detection.

---

## Expected Outcomes
- A fully functional **Streamlit-based web application** for emotion detection.  
- Accurate classification of emotions with optimized CNN models.  
- A detailed project report including system design, performance analysis, and applications.  
- Ethical considerations addressing privacy and bias mitigation in emotion detection.  

---

## Tools and Technologies
- **Programming Language**: Python  
- **Frameworks and Libraries**: 
  - [Streamlit](https://streamlit.io/) (for UI development)
  - [PyTorch](https://pytorch.org/) (for CNN implementation)
  - [Dlib](http://dlib.net/) or [Mediapipe](https://google.github.io/mediapipe/) (for facial feature extraction)  
- **Dataset**: [FER-2013](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset) (Emotion recognition dataset)
  - Available via `torchvision.datasets.FER2013`.

---

## Deliverables
1. **Streamlit Application**:  
   A fully operational web app for emotion detection.  

2. **Codebase**:  
   Well-structured and documented Python scripts for reproducibility.  

3. **Trained Models**:  
   - Fine-tuned CNN models (MobileNetV2, ResNet50, and InceptionV3) for emotion classification.  

4. **Project Report**:  
   - Comprehensive documentation of system design, methodology, results, and conclusions.  

5. **Ethical Analysis**:  
   - Discussion on privacy concerns and strategies to mitigate biases in emotion detection technology.  
