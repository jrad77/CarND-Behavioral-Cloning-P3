# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./examples/center.jpg "Center Driving"
[recover1]: ./examples/recover_1.jpg "Recovery Image"
[recover2]: ./examples/recover_2.jpg "Recovery Image"
[recover3]: ./examples/recover_3.jpg "Recovery Image"
[normal]: ./examples/recover_1.jpg "Normal Image"
[flipped]: ./examples/recover_1_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* [examples/run_track_1.mp4] and [examples/run_track_2.mp4] are videos of the car
  making it around both tracks.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is an implementation of the NVIDIA network found [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). The only modification is a dropout layer following the convolutional layers. The implementation can be found in the `build_model()` function of train.py. 

It uses five convolutional layers, followed by a dropout layer, three fully connected layers, and a final single neuron output layer. The first three convolutional layers use a 5x5 kernel and a 2x2 stride, with relu activation to introduce nonlinearities. The final two convolutional layers use a 3x3 kernel, with 1x1 stride and relu activation. The dropout layer has a drop probability of 50%.

The data is normalized in the model using a Keras lambda layer (code line 18). And is then cropped to reduce the input size and to help focus in on the important areas of the image. 

#### 2. Attempts to reduce overfitting in the model

The model uses a dropout layer after the convolutions to reduce overfitting. Additionally, it is trained and validated on different data sets to ensure that the model is not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well as driving in difficult areas a few extra times. For example, there are two turns in track one where there is just a dirt boundary on one side of the track. The car was having trouble staying on the track at that area, so I recorded a few additional examples of manual driving in that area. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the proven architecture provided by the NVIDIA team. I implemented it exactly as they had described it. The only additional steps were the normalization and cropping steps noted earlier.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and high mean squared error on the validation set. This implied that the model was overfitting to the training data. After collecting much more data and adding a dropout layer after the convolutional layers, the testing loss was greatly reduce. This implied that the model was generalizing well. The final model had a training loss of 0.0364 and a validation loss of 0.0384.

The next step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I captured additional training data in these area.

I found that the most important part of this assignment was the training data. With only a lap or two around the track, the car was not able to make it past the first turn. Adding the flipped image augmented data got the car further, however, it failed to recover when driving close to the edge of the track. Recording additional 'recovery maneuvers' improved the performance further. Finally, once I made use of the side camera data with the steering offset, the robot performed much better. Once I had collected around 100,000 sample inputs (including the augmented data) the car was able to drive around both tracks autonomously without leaving the road.

#### 2. Final Model Architecture

The final model architecture (train.py:`build_model()`) consisted of a convolution neural network with the following layers, shapes, and trainable parameters:

| Layer (type)|     Output Shape      |  Param # |
| ------------|:---------------------:| --------:|
| Lambda      | (None, 160, 320, 3)   | 0        |
| Cropping2D  | (None, 65, 320, 3)    | 0        |
| Conv2D      | (None, 31, 158, 24)   | 1824     |
| Conv2D      | (None, 14, 77, 36)    | 21636    |
| Conv2D      | (None, 5, 37, 48)     | 43248    |
| Conv2D      | (None, 3, 35, 64)     | 27712    |
| Conv2D      | (None, 1, 33, 64)     | 36928    |
| Dropout     | (None, 1, 33, 64)     | 0        |
| Flatten     | (None, 2112)          | 0        |
| Dense       | (None, 100)           | 211300   |
| Dense       | (None, 50)            | 5050     |
| Dense       | (None, 10)            | 510      |
| Dense       | (None, 1)             | 11       | 

Total trainable params: 348,219

The first three convolutional layers use a 5x5 kernel and a 2x2 stride, with relu activation to introduce nonlinearities. The final two convolutional layers use a 3x3 kernel, with 1x1 stride and relu activation.

The dropout layer has a drop probability of 50%.

The data is normalized in the model using a Keras lambda layer to center the data around 0. It is then cropped to reduce the input size and to help focus in on the important areas of the image. 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded multiple laps on track one using center lane driving. I then recorded a lap driving in the opposite direction while using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded two laps worth of the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to move back towards the center if it gets to close to the edges. These images show what a recovery looks like starting from the right side of the lane, then recovering towards the center

![alt text][recover1]
![alt text][recover1]
![alt text][recover1]

Then I repeated a similar process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help the robot generalize to turning both left and right. For example, here is an image that has then been flipped:

![alt text][normal]
![alt text][flipped]

After the collection process, I had 18252*3 input images (center, left, right). I then flipped all of those to bring the total to 109512 examples. I then preprocessed this data by normalizing the images around 0 and cropping the images.

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by  monitoring the training and validation loss. When the training loss continue to shrink, but the validation loss rose for two straight epochs I took the previous model with the lowest validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
