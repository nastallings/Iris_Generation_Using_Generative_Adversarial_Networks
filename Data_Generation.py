import os
import cv2
import numpy as np


def load_data():
    """
    This function loads data from a human iris dataset
    """

    data = []
    for filename in os.listdir("Iris Data"):
        image = cv2.imread("Iris Data/" + str(filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        data.append(image)
        # cv2.imshow("Eye Sample", image)
        # cv2.waitKey(0)
    xData = np.array(data)
    xData = np.expand_dims(xData, axis=-1)
    xData = xData.astype('float32')
    # scale data
    xData = xData / 255.0
    return xData


def generate_samples(data, numSamples):
    """
    This function generates a random number of samples from the data set to use for training
    @:param data: a dataset containing real human iris images
    @:param numSamples: the number of samples to generate
    """

    randomNumber = np.random.randint(0, data.shape[0], numSamples)
    xData = data[randomNumber]
    yData = np.ones((numSamples, 1))
    return xData, yData
