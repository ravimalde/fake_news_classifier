# Fake News Classifier

The goal of this project was to develop a classifier that can detect whether or not the headline of a news article relates to a real or fake news story. I decided to undertake this project to become more familiar with some common Natural Language Processing (NLP) practices, to work at constructing some more complicated neueral networks using Tensorflow/Keras, and to utilise Flask and Docker to deploy the model. The results were very encouraging - the best performing model, a convolutional neural network, achieved an accuracy of 97.6% on the test dataset.

- Email: ravidmalde@gmail.com
- LinkedIn: www.linkedin.com/in/ravi-malde
- Medium: www.medium.com/@ravimalde

## Table of Contents

1. [ File Descriptions ](#file_description)
2. [ Methods Used ](#methods_used)
3. [ Technologies Used ](#technologies_used)
4. [ Executive Summary ](#executive_summary)
  * [ Data Exploration ](#data_exploration)
  * [ Preprocessing ](#preprocessing)
  * [ Modelling ](#modelling)
  * [ Deployment ](#deployment)

<a name="file_description"></a>
## File Descriptions

- templates: folder containing html template to be used by the Flask app
  * form.html: html code that outlines the design of the form
- wordclouds: png images of fake and real headline wordclouds
  * fake_cloud.png
  * real_cloud.png
  * wordcloud_combined.png
- Dockerfile: file to build docker image
- app.py: python file for Flask application
- conv_net.h5: saved convolutional neural network model
- fake_news.ipynb: jupyter notebook for data preprocessing and modelling
- fake_cloud.png: wordcloud for fake headlines
- real_cloud.png: wordcloud for real headlines
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

As mentioned previously, the driver behind this project was to practice some NLP processes, construct a neural network with Tensorflow. In order to have a model to compare the neural network with, a logistic regression classifier was also developed as a baseline. The Convolutional Neural Network (CNN) performed the best on the validation and test sets achieving an accuracy of 97.6% on both. The next stage was to deploy the model using Flask and Docker to make the model accessible to others.

<a name="data_exploration"></a>
### Data Exploration

The dataset contained 23481 fake news headlines and 21417 real news headlines. This slight class imbalance was corrected for by upsamping the real news headlines to match the number of fake ones.

Wordclouds containing the top 200 most commonly featured words for both the fake and real headlines are given below. It's clear that there are some words that appear many times in both classes. Most notably and unsurprisingly, 'trump' is up there as one of the most commonly featured words for both. When taking a closer look at both of these Wordclouds, there are apparent differences between the two - from a first observation, it could be argued that the fake news headlines contain some more description and emotive language, whereas the real news headlines feature more nouns such as 'White House', 'North Korea' and 'China'.

<h5 align="center">Fake Headline Wordcloud (Left), Real Headline Wordcloud (Right)</h5>
<p align="center">
  <img src="https://github.com/ravimalde/fake_news_classifier/blob/master/wordclouds/wordcloud_combined.png" width=1000 align=middle>
</p>

<a name="preprocessing"></a>
### Data Preprocessing

The preprocessing section was split up into two parts as two different processes were required for the logistic regression baseline and the neural networks. NLP preprocessing for the logistic regression model was conducted using NLTK, for the neural networks preprocessing was done with Tensorflow.

#### Natural Language Toolkit (NLTK)

The preprocessing with NLTK comprised of the follow steps:

- Removing punctuation
- Tokenisation: seperating words within a headline
- Removing stop words: eliminating common words that do not add any meaning
- Part of speech tagging: tagging words as 'nouns', 'verbs', etc.
- Lemmatisation: restoring words to their root 
- Word vectorization: 
- TFIDF transformation

#### Tensorflow

<a name="modelling"></a>
### Modelling

#### Random Forest

#### Convolutional Neural Network

<a name="deploying_application"></a>
### Deployment

