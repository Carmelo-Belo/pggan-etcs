# pggan-etcs
The repository present the code for training a Progressive Growing GAN on an image dataset in TensorFlow Keras. It has been used for my MSc Thesis where I trained the network on the generation atmospheric variable fields of Extra-Tropical Cyclones. 

In the script PGGAN_training_comb.py is possible to modify the hyperparameters of the learning process, to change the number of channels in the input images (which need to be squared) and to define from where to upload the training set and where to save the generator architecture and weights. Scripts PGGA_architecture_comb.py and PGGAN_data_functions contains the function needed for the training.

The code is an update of the original Tensorflow implementation by Jason Bronwlee published at https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/
