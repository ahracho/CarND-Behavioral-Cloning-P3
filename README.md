# **Behaviorial Cloning Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Overview
---

This repository is for the project of **Udacity Nanodegree - Self-driving Car Engineer : Finding Lane Lines Proejct**.  It is forked from https://github.com/udacity/CarND-Behavioral-Cloning-P3).  


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

It needs us to apply everything we learned from the lecture including collecting training proper data, set model architecture, tuning hyper-parameters, and check overfitting etc. I used CNN architecture to train model, especially architecture from NVIDIA which has 5 CNN layers and 4 fully connected layers. To reduce training time, I used AWS EC2 GPU instance with Udacity AMI. 

I cannot say my final model is perfect, it still falls out from the center lane sometimes, but at least it learned how to recover to the center lane. I will refine this model to perfectly perform on autonomous mode even after submission.


## Outputs
---
This project has 3 types of outputs:
1. model.py : script used to train model. It needs argv to indicate where the training images are
~~~sh
python model.py images
~~~
2. writeup.md : writeup file that specify details on how I completed this project.
3. model-final.h5 : final model I got from the project and used to record the autonomous driving
4. final.mp4 : autonomous driving video
5. drive.py : script for test driving (I didn't modify this file)

