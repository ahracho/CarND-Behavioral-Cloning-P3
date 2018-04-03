# Load image / steering angle data from csv file
import sys
import os
import csv
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

# csv file contents
def load_csv() :
    samples = []
    with open('./' + image_folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    print("Sample size : ", len(samples))
    return samples

# Split train and validation sample
def split_train_and_validation(samples):
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print("Train sample : ", len(train_samples))
    print("Validation sample : ", len(validation_samples))

    return train_samples, validation_samples



# Using Generator so as to save memories used to train model 
import sklearn
import random

def generator(sample, batch_size=64):
    num_samples = len(sample)

    while 1:
        random.shuffle(sample)
        for offset in range(0, num_samples, batch_size):
            batch_samples = sample[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                # Center Camera
                center_name = './' + image_folder +"/IMG/"+batch_sample[0].split("\\")[-1]
                center_image = cv2.imread(center_name)
                center_image = center_image[:, :, ::-1]  # BGR to RGB
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # Flipped Image
                fliped_image = np.fliplr(center_image)
                images.append(fliped_image)
                angles.append(-center_angle)

                # Left Camera
                left_name = './' + image_folder + "/IMG/"+batch_sample[1].split("\\")[-1]
                left_image = cv2.imread(left_name)
                left_image = left_image[:, :, ::-1]  # BGR to RGB
                images.append(left_image)
                angles.append(center_angle + 0.2)

                # Right Camera
                right_name = './' + image_folder + "/IMG/"+batch_sample[2].split("\\")[-1]
                right_image = cv2.imread(right_name)
                right_image = right_image[:, :, ::-1]  # BGR to RGB
                images.append(right_image)
                angles.append(center_angle - 0.2)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Construct Network
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def create_model():
    # To save training time, I started to train model from the best model I got.
    # So if there is model.h5 file that can be re-used, I load the model, if not, I created new CNN.

    # NVIDIA network
    if not os.path.isfile("./model.h5"):
        model = Sequential()
        model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
        model.add(Lambda(lambda x: x/255.0 - 0.5))
        model.add(Convolution2D(24, 5, 5,  subsample=(2, 2), activation="relu"))
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(64, 3, 3, activation="relu"))
        model.add(Convolution2D(64, 3, 3, activation="relu"))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
    else:
        model = load_model('model.h5')
    return model



if __name__ == '__main__':
    if len(sys.argv) == 2:
        image_folder = sys.argv[1]
    else:
        image_folder = "images"

    train_samples, validation_samples = split_train_and_validation(load_csv())

    # Create Generator to save memory
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    model = create_model()
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=20, verbose=1)
    
    model.save("model.h5")
