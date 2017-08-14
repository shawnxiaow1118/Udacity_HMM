# AIND Project: Sign Language Recognition System
Udacity project on hidden Markov models
 

## Discription:
This project is to train an HMM model for sign language recognition with different methods and inputs. The main steps include:

__1. Feature selection__: preprocessing data and change coordinate to accomendate the training model. Select the proper features from the processed data.

__2. Training a model__: train a naive Gaussian HMM model to a single word. Finished different model selectors (pure cross validation, BIC and DIC) with scores to evaluate the performance of the model.

__3. Complete a recognizer__: finish the pipeline of training the model and predict from input datafile, including: process input data for trainning, select the best model for each words using different selectors and predict the words from testing dataset. Compare the performance of the recognizer.

## Requirement:
Python 3.5 or above and Jupyter notebook

## Running:
To view the ipython note book just change the directory and type:
```
jupyter notebook asl_recognizer.ipynb
```

