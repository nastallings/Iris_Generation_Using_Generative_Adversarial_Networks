import matplotlib.pyplot as plt
import numpy as np
import Data_Generation as dg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import tensorflow.keras.backend as tfback
import matplotlib.pyplot as plt
import random
import os
import cv2

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
        model.add(Convolution2D(64, (8, 8), strides=(2, 2), padding='same', input_shape=(120, 160, 1), activation="relu"))
        model.add(Dropout(0.4))
        model.add(Convolution2D(64, (8, 8), strides=(2, 2), padding='same', activation="relu"))
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
        # foundation for 15 by 20 image
        model.add(Dense(256 * 20 * 15, input_dim=self.dimensionalNoise, activation="relu"))
        model.add(Reshape((15, 20, 256)))
        model.add(Dropout(.2))

        # up-sample to 30 by 40
        model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation="relu"))
        model.add(BatchNormalization(momentum=0.8))

        # up-sample to 60 by 80
        model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation="relu"))
        model.add(BatchNormalization(momentum=0.8))

        # up-sample to 120 by 160
        model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation="relu"))
        model.add(BatchNormalization(momentum=0.8))

        # Final layer
        model.add(Convolution2D(1, (1, 1), activation='sigmoid', padding='same'))
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

    def train_GAN_model(self, trainingSet, numEpochs=100, batchSize=32):
        """
        This function trains a GAN model and updates the discriminatorModel weights.
        @:param trainingSet: Actual samples of human Iris for training
        @:param numEpochs: The number of epochs to run
        @:param batchSize: The number for the batch size
        """

        realDiscriminatorHist, syntheticDiscriminatorHist, ganHist, realDiscriminatorAcc, syntheticDiscriminatorAcc \
            = [], [], [], [], []
        for ep in range(numEpochs):
            # test the discriminatorModel by mixing real and fake samples
            xReal, yReal = dg.generate_samples(trainingSet, batchSize)
            realDiscriminatorLoss, realDiscriminatorAccuracy = self.discriminatorModel.train_on_batch(xReal, yReal)
            xGenerated, yGenerated = self.create_samples(batchSize)
            syntheticDiscriminatorLoss, syntheticDiscriminatorAccuracy \
                = self.discriminatorModel.train_on_batch(xGenerated, yGenerated)

            # prepare noise vector for GAN training
            xGan = self.generate_dimension_noise(batchSize)
            yGan = np.ones((batchSize, 1))
            GANLoss = self.GAN.train_on_batch(xGan, yGan)

            realDiscriminatorHist.append(realDiscriminatorLoss)
            syntheticDiscriminatorHist.append(syntheticDiscriminatorLoss)
            ganHist.append(GANLoss)
            realDiscriminatorAcc.append(realDiscriminatorAccuracy)
            syntheticDiscriminatorAcc.append(syntheticDiscriminatorAccuracy)

            # summarize loss on this batch
            print('epoch:%d, real_discriminator_loss=%.3f, synthetic_discriminator_loss=%.3f, gan_loss=%.3f, '
                  'real_discriminator_accuracy=%.3f, synthetic_discriminator_accuracy=%.3f'
                  % (ep + 1, realDiscriminatorLoss, syntheticDiscriminatorLoss, GANLoss,
                     realDiscriminatorAccuracy, syntheticDiscriminatorAccuracy))
            # evaluate the model performance
            if (ep + 1) % 1000 == 0:
                self.display_performance(ep, trainingSet)
            if (ep + 1) % 10000 == 0:
                # Save model
                self.Save_Models("Generative_Adversarial_Network_Model_Epoch_" + str(ep+1),
                                 "Generative_Adversarial_Network_Weights_Epoch_" + str(ep+1))

        self.plot_history(realDiscriminatorHist, syntheticDiscriminatorHist,
                          ganHist, realDiscriminatorAcc, syntheticDiscriminatorAcc)

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

    def Save_Models(self, modelName, modelWeights):
        """
        Saves the model to a JSON file
        @:param modelName: the file name for the model
        @:param modelWeights: the file name for the model weights
        """

        generativeModel_json = self.generativeModel.to_json()
        discriminatorModel_json = self.discriminatorModel.to_json()
        GAN_json = self.GAN.to_json()

        with open("Saved Models/" + modelName + "_generator", "w") as json_file_generator:
            json_file_generator.write(generativeModel_json)
        with open("Saved Models/" + modelName + "_discriminator", "w") as json_file_discriminator:
            json_file_discriminator.write(discriminatorModel_json)
        with open("Saved Models/" + modelName + "_GAN", "w") as json_file_GAN:
            json_file_GAN.write(GAN_json)

        # serialize weights to HDF5
        self.generativeModel.save_weights("Saved Models/" + modelWeights + "_generator")
        self.discriminatorModel.save_weights("Saved Models/" + modelWeights + "_discriminator")
        self.GAN.save_weights("Saved Models/" + modelWeights + "_GAN")
        print("Saved model to disk")

    def Load_Model(self):
        """
        Loads the model from a JSON file
        """

        try:
            # Load Discriminator
            disc_json_file = open(
                "Saved Models/Generative_Adversarial_Network_Model_discriminator", 'r')
            loaded_disc_model_json = disc_json_file.read()
            disc_json_file.close()
            loaded_model_disc = model_from_json(loaded_disc_model_json)
            # load weights into new model
            loaded_model_disc.load_weights("Saved Models/Generative_Adversarial_Network_Weights_discriminator")
            optimizer = Adam(lr=0.0002, beta_1=0.5)
            loaded_model_disc.compile(loss='binary_crossentropy', optimizer=optimizer)
            self.discriminatorModel = loaded_model_disc

            # Load Generator
            gen_json_file = open(
                "Saved Models/Generative_Adversarial_Network_Model_generator", 'r')
            loaded_gen_model_json = gen_json_file.read()
            gen_json_file.close()
            loaded_model_gen = model_from_json(loaded_gen_model_json)
            # load weights into new model
            loaded_model_gen.load_weights("Saved Models/Generative_Adversarial_Network_Weights_generator")
            self.generativeModel = loaded_model_gen

            # Load GAN
            gan_json_file = open("Saved Models/Generative_Adversarial_Network_Model_GAN", 'r')
            loaded_gan_model_json = gan_json_file.read()
            gan_json_file.close()
            loaded_model_gan = model_from_json(loaded_gan_model_json)
            # load weights into new model
            loaded_model_gan.load_weights("Saved Models/Generative_Adversarial_Network_Weights_GAN")
            optimizer = Adam(lr=0.0002, beta_1=0.5)
            loaded_model_gan.compile(loss='binary_crossentropy', optimizer=optimizer)
            self.GAN = loaded_model_gen

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
        print('Accuracy on Real Samples: %.0f%%, Accuracy on Generated Samples: %.0f%%'
              % (realAccuracy * 100, generatedAccuracy * 100))

        # Save sample of generated image
        plt.imsave('generated_iris_%03d.png' % (epoch + 1), xGenerated[np.random.randint(0, xGenerated.shape[0]), :, :, 0], cmap="gray")
        plt.close()

    def plot_history(self, real_d_hist, fake_2_hist, g_hist, r_a_hist, f_a_hist):
        """
        Plots the history of the model
        @:param real_d_hist: Discriminator history on real data
        @:param fake_2_hist: Discriminator history on fake data
        @:param g_hist: the GAN history
        @:param r_a_hist: Discriminator accuracy on real data
        @:param f_a_hist: Discriminator accuracy on fake data
        """

        # plot loss
        plt.subplot(2, 1, 1)
        plt.plot(real_d_hist, label='d-real')
        plt.plot(fake_2_hist, label='d-fake')
        plt.plot(g_hist, label='gen')
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Iteration")

        # plot discriminator accuracy
        plt.subplot(2, 1, 2)
        plt.plot(r_a_hist, label='acc-real')
        plt.plot(f_a_hist, label='acc-fake')
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Iteration")
        # save plot to file
        plt.savefig('plot_line_plot_loss.png')
        plt.close()

    def run_test(self):
        """
        Runs the test used on humans
        """

        data = []
        for index in range(10):
            filename = random.choice(os.listdir("Iris Data"))
            image = cv2.imread("Iris Data/" + str(filename), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
            data.append(image)
        self.Load_Model()
        xGenerated, yGenerated = self.create_samples(10)
        for image in xGenerated:
            data.append(image[:, :, 0])

        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        x = np.array(data)
        y = np.array(y)

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        x = x[indices]
        y = y[indices]

        for sample in x:
            plt.imshow(sample, cmap="gray")
            plt.show()

        print(y)

    def generate_dataset(self, numSamples):
        """
        Generates the dataset of synthetic irises
        @:param numSamples: The number of samples to create for the dataset
        """
        self.Load_Model()
        xGenerated, yGenerated = self.create_samples(numSamples)
        index = 0
        if not os.path.isdir("Synthetic Iris Dataset"):
            os.mkdir("Synthetic Iris Dataset")

        for image in xGenerated:
            # Save sample of generated image
            plt.imsave('Synthetic Iris Dataset/generated_iris_%03d.png' % index, image[:, :, 0], cmap="gray")
            index += 1
