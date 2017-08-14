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
To train your own model, replace the data in ```./data/test_words.csv```  and ```./data/train_words.csv''' file with the same format as discripted as [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php). Locate the ipython notebook inline 21 or run the following code (with feature selected):
```
model_dict = train_all_words(features, model_selector)
```
To predict the test words, loacate the ipython notebook after inline 21 or use the function:
```
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
```

The errors will be shown based on the models (or model selector) you choose by typing:
```
show_errors(guesses, test_set)
```
