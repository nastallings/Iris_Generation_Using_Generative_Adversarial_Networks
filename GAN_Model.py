import matplotlib.pyplot as plt
import numpy as np
import Data_Generation as dg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import tensorflow.keras.backend as tfback

# Sets up tensorflow GPU
tf.compat.v1.disable_eager_execution()


def _get_available_gpus():
    # global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus


class GAN_Model:
    """
    The Model Class for the GAN iris generation. Deals with creating, training, testing, and displaying the model
    """

    def __init__(self, noise):
        """
        This is the constructor for the GAN_Model.
        @:param noise: the amount of dimensional noise (also called latent space)
        """

        self.dimensionalNoise = noise
        self.generativeModel = self.generative_model()
        self.discriminatorModel = self.discriminator_model()
        self.GAN = self.gan_model()

    def discriminator_model(self):
        """
        This function creates a discriminator model for the GAN.
        @:return the discriminatorModel
        """

        model = Sequential()
        model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(80, 60, 1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def generative_model(self):
        """
        This function creates a generative model for the GAN.
        Layers look like:
            80 by 60 with depth 1
            80 by 60 with depth 120
            40 by 30 with depth 240
            20 by 15 with depth 480
        @:return the generativeModel
        """

        model = Sequential()
        # foundation for 20 by 15 image
        model.add(Dense(480 * 20 * 15, input_dim=self.dimensionalNoise, activation="relu"))
        model.add(Reshape((20, 15, 480)))
        model.add(Dropout(.4))

        # up-sample to 40 by 30
        model.add(Conv2DTranspose(240, 4, strides=(2, 2), padding='same', activation="relu"))
        model.add(BatchNormalization(momentum=0.8))

        # up-sample to 80 by 60
        model.add(Conv2DTranspose(120, 4, strides=(2, 2), padding='same', activation="relu"))
        model.add(BatchNormalization(momentum=0.8))

        # Final layer
        model.add(Conv2DTranspose(1, 4, activation='sigmoid', padding='same'))
        return model

    def gan_model(self):
        """
        This function creates a GAN model by combing the discriminatorModel and generativeModel.
        @:return the GAN model
        """

        # Ensure discriminator weights don't get updated
        self.discriminatorModel.trainable = False

        model = Sequential()
        model.add(self.generativeModel)
        model.add(self.discriminatorModel)
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model

    def train_GAN_model(self, trainingSet, epochs=100, batchSize=32):
        """
        This function trains a GAN model and updates the discriminatorModel weights.
        @:param trainingSet: Actual samples of human Iris for training
        @:param epochs: The number of epochs to run
        @:param batchSize: The number for the batch size
        """

        batchPerEpoch = int(trainingSet.shape[0] / batchSize)
        halfBatch = int(batchSize / 2)

        for ep in range(epochs):
            # enumerate batches over the training set
            for batch in range(batchPerEpoch):
                # test the discriminatorModel by mixing real and fake samples
                xReal, yReal = dg.generate_samples(trainingSet, halfBatch)
                xGenerated, yGenerated = self.create_samples(halfBatch)
                xData, yData = np.vstack((xReal, xGenerated)), np.vstack((yReal, yGenerated))
                discriminatorLoss, temp = self.discriminatorModel.train_on_batch(xData, yData)

                # prepare noise vector for GAN training
                xGan = self.generate_dimension_noise(batchSize)
                yGan = np.ones((batchSize, 1))
                GANLoss = self.GAN.train_on_batch(xGan, yGan)

                # summarize loss on this batch
                print('epoch:%d, batch: %d/%d, discriminator_loss=%.3f, gan_loss=%.3f'
                      % (ep + 1, batch + 1, batchPerEpoch, discriminatorLoss, GANLoss))
            # evaluate the model performance 1/10 of the time
            if (ep + 1) % int(epochs/10) == 0:
                self.display_performance(ep, trainingSet)

    def create_samples(self, numSamples):
        """
        This function creates samples from the noise using the generative model.
        @:param numSamples: The number of samples to use for noise creation
        @:returns: generated xData with associated yData values
        """

        generatedInput = self.generate_dimension_noise(numSamples)
        # predict outputs
        xData = self.generativeModel.predict(generatedInput)
        # create class labels
        yData = np.zeros((numSamples, 1))
        return xData, yData

    def generate_dimension_noise(self, numSamples):
        """
        This function creates dimensional noise to generate samples from.
        @:param numSamples: The number of samples to use for noise creation
        @:return an input based off of noise
        """

        # generate points in the latent space
        noise = np.random.randn(self.dimensionalNoise * numSamples)
        # reshape into a batch of inputs for the network
        noise = noise.reshape(numSamples, self.dimensionalNoise)
        return noise

    def Save_Model(self, modelName, modelWeights):
        """
        Saves the model to a JSON file
        @:param modelName: the file name for the model
        @:param modelWeights: the file name for the model weights
        """

        model_json = self.model.to_json()
        with open(modelName, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(modelWeights)
        print("Saved model to disk")

    def Load_Model(self, modelName, modelWeights):
        """
        Loads the model from a JSON file
        @:param modelName: the file name for the model
        @:param modelWeights: the file name for the model weights
        """

        try:
            json_file = open(modelName, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(modelWeights)
            loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            self.model = loaded_model
            print("Loaded model from disk")
            return 1
        except:
            return 0


    def display_performance(self, epoch, trainingSet, numSamples=100):
        """
        Displays the performance of the GAN model. Saves images of the generated irises
        @:param epoch: the epoch the model is one when the performance is requested
        @:param trainingSet: a trainingSet of real iris samples
        @:param numSamples: the number of samples to generate
        """

        # get random real samples and test them
        xReal, yReal = dg.generate_samples(trainingSet, numSamples)
        temp, realAccuracy = self.discriminatorModel.evaluate(xReal, yReal, verbose=0)

        # generate samples and test them
        xGenerated, yGenerated = self.create_samples(numSamples)
        temp, generatedAccuracy = self.discriminatorModel.evaluate(xGenerated, yGenerated, verbose=0)

        # display performance
        print('Accuracy on Real Samples: %.0f%%, Accuracy on Real Samples: %.0f%%'
              % (realAccuracy * 100, generatedAccuracy * 100))

        # Save sample of generated image
        plt.imshow(xGenerated[np.random.randint(0, xGenerated.shape[0]), :, :, 0], cmap='gray_r')
        plt.axis('off')
        plt.savefig('generated_iris_e%03d.png' % (epoch + 1))
        plt.close()


