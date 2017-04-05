import csv 
import cv2
import numpy as np

csv_file_path = '/home/jcannon/workspace/sdcnd/sim_training_data/driving_log.csv'

lines = []
with open(csv_file_path) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    img_path = line[0]
    image = cv2.imread(img_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    images.append(np.fliplr(image))
    measurements.append(-measurement)

X_train = np.array(images)
y_train = np.array(measurements)

num_examples = len(X_train)
print("Training with {} examples".format(num_examples))


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Cropping2D

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

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)

model.save('model.h5')
