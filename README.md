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
- images: png images to be used in this readme
- Dockerfile: file to build docker image
- app.py: python file for Flask application
- conv_net.h5: saved convolutional neural network model
- dense_net.h5: saved dense neural network model
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

As mentioned previously, the driver behind this project was to practice some NLP processes, construct a neural network with Tensorflow. In order to have a model to compare the neural network with, a logistic regression classifier was also developed as a baseline. The Convolutional Neural Network (CNN) performed the best on the validation and test sets achieving an accuracy of 97.6% on both. The next stage was to deploy the model using Flask and Docker to make the model accessible to others.

The following commands can be used to download and launch the application using Docker:

1. Pull the Docker image from Docker Hub by running this command in the terminal:

   ```docker pull ravimalde/fake-news-classifier:latest```

2. Run a container from this image and map it to a given port:

   ```docker run -p 1234:1234 ravimalde/fake-news-classifier:latest```

3. Open up a browser of your choice and got to localhost:1234 in the browser's address bar.

<a name="data_exploration"></a>
### Data Exploration

The dataset contained 23481 fake news headlines and 21417 real news headlines. This slight class imbalance was corrected for by upsamping the real news headlines to match the number of fake ones.

Wordclouds containing the top 200 most commonly featured words for both the fake and real headlines are given below. It's clear that there are some words that appear many times in both classes. Most notably and unsurprisingly, 'trump' is up there as one of the most commonly featured words for both. When taking a closer look at both of these Wordclouds, there are apparent differences between the two - from a first observation, it could be argued that the fake news headlines contain some more description and emotive language, whereas the real news headlines feature more nouns such as 'White House', 'North Korea' and 'China'.

<h5 align="center">Fake Headline Wordcloud (Left), Real Headline Wordcloud (Right)</h5>
<p align="center">
  <img src="https://github.com/ravimalde/fake_news_classifier/blob/master/images/wordcloud_combined.png" width=1000 align=middle>
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

The preprocessing with Tensorflow comprised of the follow steps:

- Tokenisation
- Sequencing
- Padding

<a name="modelling"></a>
### Modelling

A baseline logistic regression model was first made so that I could assess the performance of the neural networks against it. Two neural networks were then created, first a standard dense feed forward neural network, and then a convolutional neural network.

The dataset was split into a training set of approximately 38,000 instances, a validation set of 4200 instances, and a test set of 4700 instances.

#### Logistic Regression

The logistic regression configuration was selected using GridSearchCV and 5 StratifiedKFold splits. It achieved 72.4% accuracy on the training dataset and 73.1% accuracy on the validation dataset. Since there is no cost associated with false positives or false negatives, the threshold was kept at 0.5.

#### Neural Network

The structure of the first neural network constructed is given in the image below - this was the highest performing configuration after many iterations. Dropout was used to prevent the model from overfitting to the training set.

<h5 align="center">Neural Network Structure</h5>
<p align="center">
  <img src="https://github.com/ravimalde/fake_news_classifier/blob/master/images/neural_net.png" width=500 align=middle>
</p>

The image below shows the models performance changing with each training epoch. An EarlyStopping callback was used so that the model would stop training once the validation accuracy stopped performing, the model's weights would then be restored to that of the best performing epoch, which was epoch 4 (the graph starts from 0 which is actually the first epoch).

<h5 align="center">Neural Network Accuracy at Each Training Epoch</h5>
<p align="center">
  <img src="https://github.com/ravimalde/fake_news_classifier/blob/master/images/neural_net_history.png" width=600 align=middle>
</p>

#### Convolutional Neural Network (CNN)

CNNs are generally asociated with image classification tasks. This is due to their ability to detect patterns in a subspace of the image, also known as a kernel, that scans across the entirety of the image. For this same reason they are actually very good at detecting patterns in text as they can learn that certain combinations of words lead to a particular class outcome. It is essentially analysing the text using multigrams rather than unigrams, meaning multiple words are analysed together as one phrase, giving context to the individual words. For example, the phrase "It wasn't bad" should have a slightly positive sentiment, but if each word was analysed by itself this phrase would probably be classed as negative due to the high negative weighting of the word "bad". CNNs have the additional bonus of being much faster to train than Recurrent Neural Networks, which are most commonly used for NLP tasks. The image below shows the best performing CNN configuration.

<h5 align="center">Convolutional Neural Network Structure</h5>
<p align="center">
  <img src="https://github.com/ravimalde/fake_news_classifier/blob/master/images/conv_net.png" width=500 align=middle>
</p>

As before, the model's performance with each training epoch is given below and the same Early stopping callback was used. The best performing version was trained at epoch 5.

<h5 align="center">Convolutional Neural Network Accuracy at Each Training Epoch</h5>
<p align="center">
  <img src="https://github.com/ravimalde/fake_news_classifier/blob/master/images/conv_net_history.png" width=600 align=middle>
</p>

This model achieved an accuracy of 99.15% on the training set and 97.59% on the validation set. As this was the highest performing model, it was also tested on the test dataset, where it achieved an accuracy of 97.64%. This model was then saved as a file that could be loaded into the Flask application to make predictions on user's headlines.

<a name="deploying_application"></a>
### Deployment

A simple application was created using Flask and and a basic HTML script to format the page. The user can input a news headline and click on a submit button. Their submission then gets preprocessed using the same Tensorflow tokeniser and also gets padded to the same length (20 words long). The saved CNN then makes a prediction on the processed headline and returns its prediction to the user. 

To make it easy for the user to run the application a Docker image was created and pushed to Docker Hub. This can be pulled by anyone and a container can be run from it which launches the Flask application. By mapping this to a port, the application can then be easily accessed via any browser (see instructions in the executive summary above).
