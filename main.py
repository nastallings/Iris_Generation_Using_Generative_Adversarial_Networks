import GAN_Model
import Data_Generation as dg

# Create GAN model with noise dimension of 100
GAN = GAN_Model.GAN_Model(100)

# Get the trainingSet
trainingSet = dg.load_data()

# Train the GAN
GAN.train_GAN_model(trainingSet)
