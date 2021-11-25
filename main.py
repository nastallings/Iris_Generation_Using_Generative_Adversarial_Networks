import GAN_Model
import Data_Generation as dg

# Parameters
numEpochs = 1000
batchSize = 32
randomNoise = 200
modelName = "Generative_Adversarial_Network_Model"
modelWeights = "Generative_Adversarial_Network_Weights"

# Create GAN model with noise dimension of 100
GAN = GAN_Model.GAN_Model(randomNoise)

# Get the trainingSet
trainingSet = dg.load_data()

# Train the GAN
GAN.train_GAN_model(trainingSet, numEpochs, batchSize)
GAN.Save_Models(modelName, modelWeights)
