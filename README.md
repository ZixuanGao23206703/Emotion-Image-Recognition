<div align="center">
  <img src="https://github.com/ZixuanGao23206703/Emotion-Image-Recognition/blob/main/logo.png" alt="Logo" width="90" height="100">
</div>

# Emotion Image Recognition using CNN Method
</center>

![Python](https://img.shields.io/badge/python-v3.11.4+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-v1.25.1%2B-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-v2.0.3%2B-lightgrey.svg)
![Seaborn](https://img.shields.io/badge/seaborn-v0.12.2%2B-green.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3.0%2B-brightgreen.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-v3.7.2%2B-yellow.svg)
![Plotly](https://img.shields.io/badge/plotly-v5.15.0%2B-purple.svg)
![Keras](https://img.shields.io/badge/keras-v2.12.0%2B-red.svg)
![Keras Preprocessing](https://img.shields.io/badge/keras--preprocessing-v1.1.2%2B-blueviolet.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.12.1%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv--python-v4.8.0.74%2B-critical.svg)

<div align="center">
  <img src="https://github.com/ZixuanGao23206703/Emotion-Image-Recognition/blob/main/facetest.gif" alt="Animation" width="300">
</div>



## :pushpin: Overview 
This project involves building a Convolutional Neural Network (CNN) model to classify seven emotional states from human facial images: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral. The model is integrated with a webcam for real-time emotion recognition, providing practical applications in fields like healthcare, public safety, and education.

## :dart: Features  
- **Real-Time Emotion Detection**: Use your webcam to detect emotions in real time.
- **Pre-trained Model**: Skip training and use the pre-trained model for quick deployment.
- **Model Training from Scratch**: Option to train your own model using the FER2013 dataset.

## :key: Getting Started & Usage
Ensure you have Python installed, then set up the environment by installing the required packages:
```bash
pip3 install numpy
pip3 install pandas
pip3 install seaborn
pip3 install keras
pip3 install matplotlib
pip3 install plotly
pip3 install scikit-learn
pip3 install tensorflow
pip3 install opencv-python
```

### :wrench: Installation
Clone the repository:   
```bash
git clone https://github.com/ZixuanGao23206703/Emotion-Image-Recognition.git
```


### :hammer: Usage

#### Method 1: Using the Pre-trained Model
- Run the [webcam.ipynb](webcam.ipynb)    notebook to use the pre-trained model for real-time emotion recognition.


#### Method 2 : Training from Scratch
- Prepare the dataset by placing it in the appropriate folder.
- Train the model using the [main.ipynb](main.ipynb) notebook.


## :eyes:  Model Overview
- **Architecture**: The CNN model consists of three convolutional blocks with batch normalization, max pooling, and dropout layers, followed by fully connected layers.
- **Evaluation**: The model achieved an overall accuracy of 68.24% on the test set.

## :page_with_curl: License
This project is licensed under the MIT License - see the LICENSE file for details.


## :mailbox: Contact  
Email: zixuan.gao123@gmail.com  
LinkedIn: https://www.linkedin.com/in/zixuan-gia/   
GitHub: https://github.com/ZixuanGao23206703

    



