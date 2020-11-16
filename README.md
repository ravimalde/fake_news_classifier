# Fake News Classifier

The goal of this project was to develop a classifier that can detect whether or not the headline of a news article relates to a real or fake news story. I decided to undertake this project to become more familiar with some common Natural Language Processing (NLP) practices, to work at constructing some more complicated neueral networks using Tensorflow/Keras, and to utilise Flask and Docker to deploy the model. The results were very encouraging - the best performing model, a convolutional neural network, achieved an accuracy of XXX on the test dataset.

- Email: ravidmalde@gmail.com
- LinkedIn: www.linkedin.com/in/ravi-malde
- Medium: www.medium.com/@ravimalde

## Table of Contents

1. [ File Descriptions ](#file_description)
2. [ Methods Used ](#methods_used)
3. [ Technologies Used ](#technologies_used)
4. [ Executive Summary ](#executive_summary)
  * [ Preprocessing ](#preprocessing)
  * [ Modelling ](#modelling)
  * [ Deployment ](#deployment)

<a name="file_description"></a>
## File Descriptions

- templates: folder containing html template to be used by the Flask app
  - form.html: html code that outlines the design of the form
- Dockerfile: file to build docker image
- app.py: python file for Flask application
- conv_net.h5: saved convolutional neural network model
- fake_news.ipynb: jupyter notebook for data preprocessing and modelling
- requirements.txt: python dependencies needed to run the Flask application
- tokenizer.pickle: Tensorflow tokenizer transformer for data preprocessing

<a name="methods_used"></a>
## Methods Used

- Data preprocessing
- Natural Language Processing
- Machine learning
- Model Deployment

<a name="technologies_used"></a>
## Technologies Used

- Python
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- Tensorflow
- Keras
- Flask
- Docker
- HTML

<a name="executive_summary"></a>
## Executive Summary

As mentioned previously, the driver behind this project was to practice some NLP processes, construct a neural network with Tensorflow. In order to have a model to compare the neural network with, a random forest classifier was also developed as a baseline. The Convolutional Neural Network (CNN) performed better on the test and validation sets with an accuracy of XXX and XXX respectively. The next stage was to practice deploying the model using Flask and Docker. This process would open up the model, allowing for other people to use it without having to have their development environment set-up in the perfect way.

<a name="preprocessing"></a>
### Data Preprocessing

<a name="modelling"></a>
### Modelling

#### Random Forest

#### Convolutional Neural Network

<a name="deploying_application"></a>
### Deployment

