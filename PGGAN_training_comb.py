from math import sqrt
from numpy import load, asarray, zeros, ones, squeeze, savez_compressed, stack
from numpy.random import randn
from numpy.random import randint
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model
from tensorflow.keras import backend
from matplotlib import pyplot
import time
from PGGAN_architecture_comb import WeightedSum, MinibatchStdev, PixelNormalization, cramer_loss, add_discriminator_block, define_discriminator, add_generator_block, define_generator, define_composite
from PGGAN_data_functions import load_real_samples, generate_real_samples, generate_latent_points, generate_fake_samples

## Function to update the weight values in the WeightedSum layer
# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
	# calculate current alpha (linear from 0 to 1)
	alpha = step / float(n_steps - 1)
	# update the alpha for each model
	for model in models:
		for layer in model.layers:
			if isinstance(layer, WeightedSum):
				backend.set_value(layer.alpha, alpha)

## Train generator and discriminator over 1 epoch
# train a generator and discriminator
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_steps):
		# update alpha for all WeightedSum layers when fading in new blocks
		if fadein:
			update_fadein([g_model, d_model, gan_model], i, n_steps)
		# prepare real and fake samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# update the generator via the discriminator's error
		z_input = generate_latent_points(latent_dim, n_batch)
		y_real2 = ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(z_input, y_real2)
		# summarize loss on this batch
		if i == (n_steps-1):
			print('\n>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))

# scale images to preferred size
def scale_dataset(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# Function to save model_weights and plots generated by each resolution model
def summarize_performance(status, g_model, latent_dim, save_folder, n_samples=25):
    # devise name
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    '''
    # generate images
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # normalize pixel values to the range [0,1]
    X = (X - X.min()) / (X.max() - X.min())
    # plot real images
    square = int(sqrt(n_samples))
    for i in range(n_samples):
      pyplot.subplot(square, square, 1 + i)
      pyplot.axis('off')
      pyplot.imshow(squeeze(X[i]))
    # save plot to file
    filename1 = 'plot_%s.png' % (name)
    pyplot.savefig(save_folder+filename1)
    pyplot.close()
    '''
    # save the weight of the generator model
    filename_weights = 'model_weigths_%s.h5' % (name)
    g_model.save_weights(save_folder+filename_weights)
    print('\n>Saved: %s' % (filename_weights))

## Function to try on all the epochs
# train the generator and discriminator
def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch, save_folder):
	# fit the baseline model
	g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
	# scale dataset to appropriate size
	gen_shape = g_normal.output_shape
	scaled_data = scale_dataset(dataset, gen_shape[1:])
	# train normal or straight-through models
	train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
	summarize_performance('tuned', g_normal, latent_dim, save_folder)
	# process each level of growth
	for i in range(1, len(g_models)):
		# retrieve models for this level of growth
		[g_normal, g_fadein] = g_models[i]
		[d_normal, d_fadein] = d_models[i]
		[gan_normal, gan_fadein] = gan_models[i]
		# scale dataset to appropriate size
		gen_shape = g_normal.output_shape
		scaled_data = scale_dataset(dataset, gen_shape[1:])
		print('Scaled Data', scaled_data.shape)
		# train fade-in models for next level of growth
		train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
		summarize_performance('faded', g_fadein, latent_dim, save_folder)
		# train normal or straight-through models
		train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
		summarize_performance('tuned', g_normal, latent_dim, save_folder)

## TRAINING OF THE MODEL
# number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
n_blocks = 5
# learning rate
lr = 10e-5
# Adam optimizer hyperparameters
b_1 = 0.0
b_2 = 0.99
eps = 10e-8
# size of the latent space
latent_dim = 128
# number of batches for each growth phase
n_batch_res = [16, 16, 16, 8, 8]
n_batch_mul = 2
n_batch = [element * n_batch_mul for element in n_batch_res]
# number of epochs for each growth phase
# 10 epochs == 500K images per training phase
n_epochs_res = [2, 4, 4, 5, 5]
n_epochs_mul = 8
n_epochs = [element * n_epochs_mul for element in n_epochs_res]

# Define models & input shape, the input shape depends on the training set
in_shape = (4, 4, 3)
channels = in_shape[-1]
d_models = define_discriminator(n_blocks, lr, b_1, b_2, eps, input_shape=in_shape)
g_models = define_generator(latent_dim, n_blocks, channels)
# Define composite models
gan_models = define_composite(d_models, g_models, lr, b_1, b_2, eps)

# load image data
# Here the training set need to be imported, eventual data transformation needs to be added by user
dataset = load_real_samples('')

# Save folder definition for where we want to save model and weights at the end of training
save_folder = ''

# train the model
start_time = time.time()
train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch, save_folder)
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

## Save highest resolution tuned generator and discriminator model architecture
g_model = g_models[-1][0]
gen_shape = g_model.output_shape
name = '%03dx%03d-tuned.h5' % (gen_shape[1], gen_shape[2])
g_model.save(save_folder+name)
d_model = d_models[-1][0]
d_name = 'disc_%03dx%03d-tuned.h5' % (gen_shape[1], gen_shape[2])
d_model.save(save_folder+d_name)