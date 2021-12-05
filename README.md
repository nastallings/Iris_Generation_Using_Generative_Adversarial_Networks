# Iris_Generation_Using_Generative_Adversarial_Networks
This is my final project for deep learning. The code for the repository can be found here: https://github.com/nastallings/Iris_Generation_Using_Generative_Adversarial_Networks

This project is completed in three files: GAN_MODEL.py, Data_Generateion.py, and main.py.

Data_Generateion.py deals will collecting and preparing the data from the Iris Data folder. This should not be touched unless the images need to resized to something different. 

GAN_MODEL.py deals with the creation, testing, and generation of fake iris samples. The file has three models in it, along with the ability to save and load older models.

main.py is what needs to be run to test the code. It has essential parameters that can be changed to impact the code. numEpochs is the number of epochs to run during training. batchSize is the size of batches during training. randomNoise is the size of the random noise used to generate iris samples. train is a variable set to false. Set it to true if you want to train a new model. This will take a long time. test is set to false. Set it to true if you want to run a comparison test for humans. data_set is set to true. This generates a set of 100 syntehtic iris samples and saves them to the Synthetic Iris Dataset folder. The number of images generated can be changed on line 31. 

The saved model is in Saved Models. This should not be touched as it runs the risk of corrupting the trained model. Due to the large file size, the code was pushed with git LGS. More information can be found here: https://git-lfs.github.com/

Sample_Generated_Iris.png contains an image of a generated iris using the GAN.
Performance-over-time-plot.png shows the performace of the GAN over the 150000 iterations. 
