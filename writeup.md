# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./capture_images/iterative_process.png "Iterative Process"
[image2]: ./capture_images/curve_1.png "curve 1"
[image3]: ./capture_images/curve_2.png "curve 2"
[image4]: ./capture_images/cnn-architecture.png "CNN Architecture"
[image5]: ./capture_images/driving_center.jpg "Drive Center"
[image6]: ./capture_images/recovery_1.jpg "Recover"
[image7]: ./capture_images/recovery_2.jpg "Recover"
[image8]: ./capture_images/recovery_3.jpg "Recover"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
~~~sh
python drive.py model.h5
~~~

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the CNN. You can specify folder containing training images as option (Default is images folder).
~~~sh
python model.py images
~~~

The file shows the pipeline I used for training and validating the model. When I construct model, I re-used previous best model I got so as to save training time.



### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I started from NVIDIA model which consists of a convolution neural network; 3 layers with 5x5 filter sizes / depths 24, 36, 48 and 2 layers with 3x3 filter size / depth 64. (model.py lines 92-97) I used AWS EC2 instances for reducing training time.  

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 91).   
To reduce image size, cropped upper part of the image(mountain, sky etc.) using Cropping2D (code line 90).  

After Convolutional layer, I flattened dataset and used fully connected layers (code line 100-103).  

I neglected the important tip which is written at the end of readme file. Loaded image data has different format between cv2 and drive.py. After bunch of trials and errors, I still failed to make the vehicle run stably, and I thought everything I can do to improve my model. After wasting so much time, I just found out this information. 

I added code for change BGR format to RGB right after loading images (code line 50, 63, 70). After applying that, model improved dramatically.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 95, 99).   

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 22-27).  

#### 3. Model parameter tuning

The model used an adam optimizer, and loss function as MSE (code line 104).  
I used image from center camera, flipped image, and image from left and right camera, so I set batch size as 64 so that operation speed cannot be slow down too much.  

Epoch is set between 10 and 20. Because I followed iterative process, I wanted to reduce the training time, so set epoch not too big.  

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. And I differed vehicle speed(slow and fast version) in training session, expected they may provide different information. I used mouse to control vehicle since it provides more fine angle than keyboard.  

While testing with drive.py script, I checked in which part of the lap vehicle drives unstable, and added more training data for that course.  

![Iterative Process][image1]  



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At the very first, I started set architecture just with a few of fully connected layers. I thought if layers are large enough to contain features, then it can deliver desirable result. But unlike my expectation, fully connected network didn't work well in this case (or maybe network was not complex enough).  

I changed architecture to NVIDIA model which guarantees better performance. I set 3 CNN layers with 5x5 filter size and depth 24, 36, 48 and 2 CNN layers with 3x3 filter size and depth 64. After CNN layers, 4 fully connected layers are used. I used dropout layers to prevent model to be overfitted to the training data.  

I split data set to train and validation set to check if model is overfitted to training data or not. Since I used dropout layers overfitting problem rarely happened (Over the iteration loss for validation data kept decresing).  

After training model, I ran the simulator to see how it works. There were a few spots where the vehicle fell off the track. In my case, turning left curve right after crossing the bridge, and sharp right curve were the problem. Data for driving on the curves were relatively less than those for driving straight, so I added more image data to train how to drive on curves.  

![Curve 1][image2]  
![Curve 2][image3]  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

create_model funtion(code line 86) describes the final model architecture. I used NVIDIA architectual after several trials. As I mentioned above, there are 3 CNN layers with 5x5 filter size and depth 24, 36, 48 and 2 CNN layers with 3x3 filter size and depth 64. After CNN layers, 4 fully connected layers are used. I used dropout layers to prevent model to be overfitted to the training data.  
I cropped upper part of the image to remove unnecessary image data(like sky, mountains) and used Lambda to normalize image data.  

Here is image of my model architecture.  
![Architecture][image4]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving, I drove the vehicle relatively slowly (between 8~10 MPH) so that I can control angles much elaboratly. Here is an example image of center lane driving:

![Center][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to get back to center lane when it is not. Here are some images I recorded from recovery session. For recovery session, I speed up a little bit, since with higher speed I could generate more cases vehicle need to recover to center lane. 

![Recover][image6]
![Recover][image7]
![Recover][image8]

To augment the data sat, I also used flipped images and images from left/right cameras. In this way, I could make use of much more image data than I captured. Since image format were different with cv2.imread() and drive.py, and reversed the image data right after it was loaded.

At first, I started to train my model with 20K training data, and during test, I captured more data for the points vehicle kept falling off. Finally, I got around 40K training data to complete my model. 
![Iterative][image1]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Thanks to dropout method, overfitting was not the problem most of the time. I used an adam optimizer so that manually training the learning rate wasn't necessary.


#### Retrospect
It was my very first time adjusting CNN architecture to train model, so I wasted so much time to figure out what to tune to improve my model during this project. 

Things I learned from this project:
- **Collecting proper dataset is important** : When I used keyboard to control vehicle, angles calculated in autonomous mode were too big and vehicle was easily fall off from the road. Furthermore, just with center lane driving data, vehicle didn't drive well once it was not on the center lane (it didn't know how to recover).
- **How to train time/cost efficiently** : At first, I didn't come up with I can re-use the trained model to refine it. So whenever I need to train model, I started it from the scratch over and over again, it was waste of time and cost(AWS remaining credit were decreasing :( ). Doing project I need to think of an idea to reduce time/cost consumption.
- **I need to learn from what people have done already** : I usually like to do all by myself from the scratch. In this way, I can experience more from trials and erros, and learn much more. But sometimes learning from what has been already achieved is better (That is the spirit of feature extraction!). The biggest mistake I have done during this project was that I didn't listen to what others say, and ask help when I was stuck in problems. If I found out BGR/RGB problem eariler, I could have saved time and effort and go to the next level.
