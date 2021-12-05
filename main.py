import GAN_Model
import Data_Generation as dg

# Parameters
numEpochs = 150000
batchSize = 48
randomNoise = 100
modelName = "Generative_Adversarial_Network_Model"
modelWeights = "Generative_Adversarial_Network_Weights"
train = False
test = False
data_set = True


# Create GAN model with noise dimension of 100
GAN = GAN_Model.GAN_Model(randomNoise)

# Get the trainingSet
trainingSet = dg.load_data()

if train:
    # Train the GAN
    GAN.train_GAN_model(trainingSet, numEpochs, batchSize)

if test:
    # Test the GAN
    GAN.run_test()

if data_set:
    # Generate Data Base
    GAN.generate_dataset(100)

