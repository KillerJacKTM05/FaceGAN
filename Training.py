# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 22:36:01 2023

@author: doguk
"""

import os
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Reshape, Concatenate, Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose

# Load partitioning information
partition_df = pd.read_csv('list_eval_partition.csv')
# Load attributes
attributes_df = pd.read_csv('list_attr_celeba.csv')
# image folderpath
folderPath = './img_align_celeba'

# Data Preprocessing
def load_image(image_path, labels):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = 2 * (img / 255.0) - 1  # Normalize to [-1, 1]
    return img, labels

class AdaIN(Layer):
    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 4:
           raise ValueError("Expected input shape to be 4D (batch_size, height, width, channels)")
           
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],), initializer='ones', trainable=True)

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        return self.gamma * (inputs - mean) / tf.sqrt(var + 1e-7) + self.beta

def create_generator(latent_dim, num_classes):
    noise_input = Input(shape=(latent_dim,))
    labels_input = Input(shape=(num_classes,))

    labels_embedding = Dense(4 * 4 * 128)(labels_input)
    labels_embedding = Reshape((4, 4, 128))(labels_embedding)

    noise_layer = Dense(4 * 4 * 128)(noise_input)
    noise_layer = Reshape((4, 4, 128))(noise_layer)

    merged_input = Concatenate()([noise_layer, labels_embedding])

    # Initial Convolution
    x = Conv2D(128, (4, 4), padding='same')(merged_input)
    adain_layer = AdaIN()
    x = adain_layer(x)
    x = LeakyReLU()(x)

    # Upsampling blocks
    for _ in range(4):  # Increase this for higher resolution
        x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
        adain_layer = AdaIN()
        x = adain_layer(x)
        x = LeakyReLU()(x)

    # Final Convolution to produce image
    x = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same')(x)
    output_img = layers.Activation('tanh')(x)  # Normalize to [-1, 1]

    model = Model([noise_input, labels_input], output_img)
    return model

def create_discriminator(img_shape, num_classes):
    img_input = Input(shape=img_shape)
    labels_input = Input(shape=(num_classes,))

    labels_embedding = Dense(np.prod(img_shape))(labels_input)
    labels_embedding = Reshape(img_shape)(labels_embedding)

    merged_input = Concatenate()([img_input, labels_embedding])

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(merged_input)
    x = LeakyReLU()(x)

    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(1, (4, 4), strides=(1, 1), padding='same')(x)

    model = Model([img_input, labels_input], x)
    return model

def wgan_loss(y_true, y_pred):
    y_true = K.expand_dims(K.expand_dims(K.expand_dims(y_true, -1), -1), -1)
    return K.mean(y_true * y_pred)

def calculate_gradient_penalty(averaged_samples, real_labels, fake_labels, discriminator):
    # Calculate the predictions for the averaged samples
    averaged_preds = discriminator([averaged_samples, tf.multiply(0.5, tf.add(tf.cast(real_labels, tf.float32), tf.cast(fake_labels, tf.float32)))])
    
    # Calculate the gradients of the predictions with respect to the averaged samples
    with tf.GradientTape() as tape:
        tape.watch(averaged_samples)
        averaged_preds = discriminator([averaged_samples, tf.multiply(0.5, tf.add(tf.cast(real_labels, tf.float32), tf.cast(fake_labels, tf.float32)))])
        
    gradients = tape.gradient(averaged_preds, averaged_samples)
    
    # Calculate the gradient's L2 norm
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    
    # Compute the gradient penalty
    gradient_penalty = K.square(1 - gradient_l2_norm)
    
    return gradient_penalty

def custom_binary_crossentropy(y_true, y_pred):
    y_pred = K.mean(y_pred, axis=[1, 2, 3])  # Average over the spatial dimensions
    return K.binary_crossentropy(y_true, y_pred)


# Hyperparameters
latent_dim = 100
gradient_penalty_weight = 10  # Lambda in WGAN-GP loss
num_classes = attributes_df.shape[1] - 1
img_shape = (128, 128, 3)
gen_lr = 0.0001
disc_lr = 0.0002

if __name__ == "__main__":
    #Allow gpu options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    gan_epochs = int(input("Enter number of epochs for training GAN: "))
    batch_size = int(input("Enter batch size for training GAN: "))
    
    # Load images
    # Create a list of image paths
    image_paths = [os.path.join(folderPath, fname) for fname in partition_df['image_id'].values]
    labels = attributes_df.iloc[:, 1:].values
   
    # Combine partition, image_paths, and labels into a single DataFrame
    combined_df = partition_df.copy()
    combined_df['image_path'] = image_paths
    combined_df = pd.merge(combined_df, attributes_df, on='image_id')
   
    # Filter based on partition
    train_df = combined_df[combined_df['partition'] == 0]
    val_df = combined_df[combined_df['partition'] == 1]
    test_df = combined_df[combined_df['partition'] == 2]
   
    # Randomly sample 1000 rows from the training DataFrame
    train_subset = train_df.sample(n=1000)
   
    # Create separate lists for image_paths and labels for the subset
    train_image_paths_subset = train_subset['image_path'].values.tolist()
    train_labels_subset = train_subset.iloc[:, 3:].values  # Assuming attributes start from column 5
   
    # Create separate datasets
    train_dataset_subset = tf.data.Dataset.from_tensor_slices((train_image_paths_subset, train_labels_subset))
    train_dataset_subset = train_dataset_subset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # Randomly sample 200 rows from the validation DataFrame
    val_subset = val_df.sample(n=200)

    # Create separate lists for image_paths and labels for the subset
    val_image_paths_subset = val_subset['image_path'].values.tolist()
    val_labels_subset = val_subset.iloc[:, 3:].values  # Assuming attributes start from column 5

    # Create separate datasets
    val_dataset_subset = tf.data.Dataset.from_tensor_slices((val_image_paths_subset, val_labels_subset))
    val_dataset_subset = val_dataset_subset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
    
    # Initialize models
    generator = create_generator(latent_dim, num_classes)
    discriminator = create_discriminator(img_shape, num_classes)

    # Compile models
    gen_optimizer = tf.keras.optimizers.Adam(gen_lr, 0.5)
    disc_optimizer = tf.keras.optimizers.Adam(disc_lr, 0.5)
    discriminator.compile(optimizer=disc_optimizer, loss=wgan_loss)
    generator.compile(optimizer=gen_optimizer, loss=custom_binary_crossentropy)
    
    # Initialize ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    reduce_lr.set_model(generator)
    
    # Placeholder for storing losses
    disc_losses = []
    gen_losses = []

    # Training loop
    for epoch in range(gan_epochs):
        print(f"Epoch {epoch + 1}/{gan_epochs} starting..")        
        for real_imgs, real_labels in  train_dataset_subset:        
            # Generate fake images using the generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_labels = np.random.choice([-1, 1], size=(batch_size, num_classes))
            fake_imgs = generator.predict([noise, fake_labels])
            
            # Train the discriminator
            real = np.ones((batch_size, 1))
            fake = -np.ones((batch_size, 1))
        
            # Calculate averaged samples
            averaged_samples = 0.5 * (real_imgs + fake_imgs)
    
            # Calculate gradient penalty (this is a simplified example; you'll need to implement this)
            gradient_penalty = calculate_gradient_penalty(averaged_samples, real_labels, fake_labels, discriminator)
            
            d_loss_real = discriminator.train_on_batch([real_imgs, real_labels], real)
            d_loss_fake = discriminator.train_on_batch([fake_imgs, fake_labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) + gradient_penalty_weight * gradient_penalty
        
            # Train the generator
            g_loss = generator.train_on_batch([noise, fake_labels], real)
        
            # Store losses for visualization
            disc_losses.append(d_loss)
            gen_losses.append(g_loss)
            
        for val_imgs, val_labels in val_dataset_subset:
            # Calculate validation loss   
            val_noise = np.random.normal(0, 1, (batch_size, latent_dim))
            val_fake_labels = np.random.choice([-1, 1], size=(batch_size, num_classes))
            val_fake_imgs = generator.predict([val_noise, val_fake_labels])

            val_real = np.ones((batch_size, 1))
            val_fake = -np.ones((batch_size, 1))

            val_d_loss_real = discriminator.evaluate([val_imgs, val_labels], val_real, verbose=0)
            val_d_loss_fake = discriminator.evaluate([val_fake_imgs, val_fake_labels], val_fake, verbose=0)
            val_d_loss = 0.5 * np.add(val_d_loss_real, val_d_loss_fake)
    
        # Apply ReduceLROnPlateau
        reduce_lr.on_epoch_end(epoch, logs={'val_loss': val_d_loss})
    
        # Visualization and model-saving every 10% of total epochs
        if epoch % (gan_epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{gan_epochs} completed. D Loss: {d_loss} G Loss: {g_loss}")
        
            # Generate some sample images and visualize them
            noise = np.random.normal(0, 1, (1, latent_dim))
            sampled_labels = np.random.choice([-1, 1], size=(1, num_classes))
            gen_imgs = generator.predict([noise, sampled_labels])
        
            # Rescale images from [-1, 1] to [0, 1]
            gen_imgs = 0.5 * gen_imgs + 0.5
        
            plt.imshow(gen_imgs[0])
            plt.title(f"Epoch {epoch}")
            plt.axis('off')
            plt.show()
        
            # Save the generator and discriminator models
            save_path = f"./models/epoch_{epoch}"
            os.makedirs(save_path, exist_ok=True)
            generator.save(os.path.join(save_path, "generator.h5"))
            discriminator.save(os.path.join(save_path, "discriminator.h5"))

    # Plotting the losses
    plt.figure()
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.plot(gen_losses, label="Generator Loss")
    plt.legend()
    plt.show()