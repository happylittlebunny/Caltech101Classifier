# Skymind Coding Assignment 

[Coding Assignment Description](https://www.zepl.com/viewer/notebooks/bm90ZTovL2Nyb2NrcG90dmVnZ2llcy9hZjAyZmEzOTk3M2Y0NmRhODFhM2Y0OGMzNmU0OTI5NC9ub3RlLmpzb24) <br />
This project uses [Deeplearning4J](https://github.com/deeplearning4j/deeplearning4j) to recognize the different categories in [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) dataset.

## About Caltech 101

The data is now contains 102 categories, so I removed BACKGROUND_Google and used the left 101 categoies

## How I trained the model

Inspired by https://deeplearning4j.org/transfer-learning, I used VGG16 pretrained model with fine tune and 10 epochs and got accuracy and precision higher than 70%
```
==========================Scores========================================
 # of classes:    101
 Accuracy:        0.7843
 Precision:       0.8082
 Recall:          0.7848
 F1 Score:        0.7828
Precision, recall & F1: macro-averaged (equally weighted avg. of 101 classes)
========================================================================
```
## How to run the project 

* IntelliJ IDE:<br />
This is a maven project. It's developed in IntelliJ. The project can be loaded and run in IntelliJ.
When run in IntelliJ, under "Run"->"Edit Configurations", update following:
  * VM options: -Xms8g -Xmx8g
  * Program arguments: <data path ie: C:\\Users\\Yuyi\\Desktop\\bigdata2017Fall\\skymind\\data\\train>

* Command line:<br />
The Caltech101Classifier-1.0-SNAPSHOT-jar-with-dependencies.jar is under target folder.<br />
Run Caltech101Classifier-1.0-SNAPSHOT-jar-with-dependencies.jar:<br />
Go to the jar directory and run:
```
java -jar -Xms8g -Xmx8g Caltech101Classifier-1.0-SNAPSHOT-jar-with-dependencies.jar <data path>
```

## My environment 

OS: Windows 7 <br />
RAM: 16G <br />
NO GPU <br />

I trained this model in my environment for about 5 hours

## Thoughts about this project

I was very excited to get this coding assignment. I am new to deep learning and it's a perfect chance to learn and experience deep learning and at the same time try the famous deeplearning4j libray. However after few attempts, I thought it's an impossible mission. I started it with 3 classes and a very simple neural network with 3 convolution layers,3 sampling layers, 1 dense layer and 1 dropout layer, 10 epochs. The accuracy can reach 70% for 3 classes and took about 20 minutes to run. But when classes increased, the simple neural network accuracy dropped a lot. And training model took much longer to run. I only have CPU available. It's very time consuming to tune a model for 101 classes. I checked deeplearning4j documents and other online resources and figured out I can use VGG16 pretrained model with fine tune instead of training a model from scratch. VGG16 is the perfect model for image recognization. I checked deeplearning4j's example code on how to use pretrained model. First time I used VGG16, I ran out of memory. Then I increased JVM memory to 8G and fixed the memory issue. The first example I tried with VGG16 took 2 hours for each epoch, 10 epochs is about 20 hours which is too long for me to wait for a result. After a few attempts, I found the model I am using now, it uses frozen layers which only used in test mode. The time for each epoch using current model is about half an hour, 10 epochs is about 5 hours and the result is remarkable! <br/>

Yuyi Zhou
