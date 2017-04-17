import csv 
import cv2
import numpy as np
import argparse

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Cropping2D
from keras.callbacks import ModelCheckpoint


def get_training_data(csv_file_path):
    '''
    Get the saved training data, as well as the generated 'augmented' training
    data. The generated data is created by flipping all saved images and
    negating the steering angle. The left and right cameras are used as well
    with a hand tuned steering_correction
    '''
    lines = []
    with open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []

    def add_image(image, steering):
        images.append(image)
        measurements.append(steering)
        images.append(np.fliplr(image))
        measurements.append(-steering)


    for line in lines:
        img_path = line[0]
        image = cv2.imread(img_path)
        steering_center = float(line[3])
        add_image(image, steering_center)
        
        correction = 0.35 # this is a parameter to tune
        steering_left = steering_center + correction
        image_left = cv2.imread(line[1])
        add_image(image_left, steering_left)
        
        steering_right = steering_center - correction
        image_right = cv2.imread(line[2])
        add_image(image_right, steering_right)


    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train

def build_model():
    '''
    Build up the model and return it.
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))

    # Conv Layer 1
    model.add(Convolution2D(filters=24, kernel_size=5, strides=2, activation='relu'))

    # Conv Layer 2
    model.add(Convolution2D(filters=36, kernel_size=5, strides=2, activation='relu'))

    # Conv Layer 3
    model.add(Convolution2D(filters=48, kernel_size=5, strides=2, activation='relu'))

    # Conv Layer 4
    model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu'))

    # Conv Layer 5
    model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu'))

    # Flatten out and add the fully connected layers
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def train(model, X_train, y_train, epochs):
    '''
    Perform the training and validation steps.

    model - the compiled keras model
    X_train - the unnormalized training inputs
    y_train - the target steering angle for each input
    epochs - the number of epochs to train for
    '''
    filepath="model-{epoch:d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, callbacks=[checkpoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDC Trainer')
    parser.add_argument('--training_csv', 
                        type=str, 
                        default='/home/jcannon/workspace/sdcnd/sim_training_data/driving_log.csv',
                        help='path to training data csv')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=5, 
                        help='number of epochs to train for')

    args = parser.parse_args()

    X_train, y_train = get_training_data(args.training_csv)
    print("Training with {} examples".format(len(X_train)))

    model = build_model()
    print(model.summary())

    train(model, X_train, y_train, args.epochs)
