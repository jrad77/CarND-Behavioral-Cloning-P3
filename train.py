import csv 
import cv2
import numpy as np
import argparse

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Cropping2D


def get_training_data(csv_file_path):
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

    for i in range(epochs):
        print("Training iteration", i)
        model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)
        model_name = 'model-{}.h5'.format(i)
        model.save(model_name)
        print("Saved as {}".format(model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDC Trainer')
    parser.add_argument('--training_dir', 
                        type=str, 
                        default='/home/jcannon/workspace/sdcnd/sim_training_data/driving_log.csv',
                        help='path to training data')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=5, 
                        help='number of epochs to train for')

    args = parser.parse_args()

    X_train, y_train = get_training_data(args.training_dir)
    print("Training with {} examples".format(len(X_train)))

    model = build_model()

    train(model, X_train, y_train, args.epochs)
