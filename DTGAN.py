# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:58:31 2023

@author: doguk
"""

import os
import math
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy

# Directory setup
data_dir_raf = "./RAF-DB"
save_dir = './saved_models/'
data_dir_clb = "./Celeb-A/Celeb-A"
clb_feat_dir = "./Celeb-A/list_attr_celeba.csv"

# Image size and features setup
img_size = (128, 128)  # Updated image size
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_to_label = {emotion: i for i, emotion in enumerate(emotions)}
selected_features = [8, 9, 11, 17, 20, 39]  # for Celeb-A

# Initialization of arrays to store images and labels
raf_images = []
raf_labels = []
celebA_images = []
celebA_labels = []

def load_celebA_features(filename, selected_features):
    """Load and preprocess Celeb-A features from CSV file."""
    df = pd.read_csv(filename, header=0)

    # Select the required columns (first column for image names and selected features)
    df_selected = df.iloc[:, [0] + [f + 1 for f in selected_features]]

    # Convert -1 to 0 in the feature columns
    feature_columns = df_selected.columns[1:]  # Exclude the first column (image names)
    df_selected.loc[:, feature_columns] = df_selected.loc[:, feature_columns].replace(-1, 0)

    return df_selected


def LoadCeleb():
    """Load Celeb-A images and corresponding features."""
    celebA_features = load_celebA_features(clb_feat_dir, selected_features)
    for _, row in celebA_features.iterrows():
        img_name = row['image_id']  # Use the 'image_id' column for the image filename
        img_path = os.path.join(data_dir_clb, img_name)
        img = Image.open(img_path).resize(img_size)
        celebA_images.append(np.array(img))
        features = row[1:].values  # Get the features, excluding 'image_id'
        celebA_labels.append(features)


def LoadRaf():
    """Load RAF-DB images and corresponding labels."""
    for i, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir_raf, f"{i}_{emotion}")
        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            img = Image.open(img_path).resize(img_size)
            raf_images.append(np.array(img))
            raf_labels.append(emotion_to_label[emotion])
            
def get_data_loader(images, labels, batch_size, shuffle=True):
    # Shuffle the dataset
    if shuffle:
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]

    # Check if the dataset size is divisible by the batch size
    if len(images) % batch_size != 0:
        # If not, find the number of excess images
        excess = len(images) % batch_size
        # Remove the excess images so that it's divisible
        images = images[:-excess]
        labels = labels[:-excess]

    # Yield batches
    for i in range(0, len(images), batch_size):
        yield images[i:i + batch_size], labels[i:i + batch_size]
        
def DisplayImages(size, images):
    plt.figure(1, figsize=(10, 10))
    for i in range(size):
        plt.subplot(int(math.sqrt(size)), int(math.sqrt(size)), i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()
    
def build_discriminator(img_shape, num_raf_labels, num_celeba_labels):
    # Image input
    img_input = layers.Input(shape=img_shape)
    # Label input (including dataset flags)
    label_input = layers.Input(shape=(num_raf_labels + num_celeba_labels + 2,))  # +2 for dataset flags

    # Convolutional layers
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(img_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, label_input])

    # Source classification (real/fake)
    validity = layers.Dense(1, activation='sigmoid')(x)

    # Class classification (label vector)
    label = layers.Dense(num_raf_labels + num_celeba_labels + 2, activation='softmax')(x)  # +2 for dataset flags

    # Model definition
    model = Model(inputs=[img_input, label_input], outputs=[validity, label])
    return model 

def residual_block(input_tensor, filters):
    """A Residual Block as defined in StarGAN."""
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.Add()([x, input_tensor])
    return x

def build_generator(z_dim, img_shape, num_raf_labels, num_celeba_labels):
    # Noise input
    noise_input = layers.Input(shape=(z_dim,))
    
    # Label input (including dataset flags)
    label_input = layers.Input(shape=(num_raf_labels + num_celeba_labels + 2,))  # +2 for dataset flags
    
    # Concatenate noise and label
    combined_input = layers.Concatenate()([noise_input, label_input])
    
    # Dense layer to prepare input for up-sampling
    x = layers.Dense(128 * (img_shape[0] // 4) * (img_shape[1] // 4))(combined_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((img_shape[0] // 4, img_shape[1] // 4, 128))(x)
    
    # Down-sampling blocks
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Bottleneck blocks (residual blocks)
    for _ in range(6):
        x = residual_block(x, 512)
    
    # Up-sampling blocks
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(256, kernel_size=3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(128, kernel_size=3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(128, kernel_size=3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(64, kernel_size=3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Output layer with tanh activation
    img_output = layers.Conv2DTranspose(img_shape[2], kernel_size=7, padding='same', activation='tanh')(x)
    
    # Define model
    model = Model(inputs=[noise_input, label_input], outputs=img_output)
    return model

def encode_labels(raf_label, celeba_feature_vector, flag):
    # Create a one-hot encoding for RAF-DB label
    raf_encoded = np.zeros(len(emotions))
    if raf_label is not None:
        raf_encoded[raf_label] = 1
    
    # Extend the Celeb-A feature vector with zeros for RAF-DB emotions
    extended_celeba_vector = np.zeros(len(emotions)) if flag == 1 else celeba_feature_vector
    
    # Combine the RAF-DB label and Celeb-A features into one vector
    combined_label = np.concatenate((raf_encoded, extended_celeba_vector), axis=None)
    
    # Add the dataset flag at the end of the label vector
    dataset_flag = [1, 0] if flag == 1 else [0, 1]  # [Celeb-A, RAF-DB]
    combined_label_with_flag = np.concatenate((combined_label, dataset_flag), axis=None)
    
    return combined_label_with_flag

# Assuming discriminator output is [src, cls]
# src: source classification (real/fake)
# cls: class classification (label vector)
binary_crossentropy = BinaryCrossentropy(from_logits=True)
categorical_crossentropy = SparseCategoricalCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output, real_labels, fake_labels, flag):
    # Real/fake loss
    real_loss = binary_crossentropy(tf.ones_like(real_output[0]), real_output[0])
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output[0]), fake_output[0])
    src_loss = (real_loss + fake_loss) / 2

    # Apply mask to class predictions based on dataset flag
    mask = tf.cast(tf.expand_dims(real_labels[:, -2:], axis=-1), tf.float32)  # [batch_size, 1, 2]
    real_class_predictions = real_output[1] * mask
    fake_class_predictions = fake_output[1] * mask
    
    # Class loss - only for the active dataset
    real_class_loss = categorical_crossentropy(real_labels, real_class_predictions)
    fake_class_loss = categorical_crossentropy(fake_labels, fake_class_predictions)
    cls_loss = (real_class_loss + fake_class_loss) / 2
    
    total_loss = src_loss + cls_loss
    return total_loss

# Main execution
flag = int(input("Which dataset will be trained (1 for Celeb-A, else RAF-DB): "))
epochs = int(input("Enter the number of epochs: "))
batch_size = int(input("Enter the batch size: "))
img_shape = (128, 128, 3)  # Image dimensions (height, width, channels)
z_dim = 100  # Dimension of the noise vector

if flag == 1:
    LoadCeleb()
    celebA_images = np.array(celebA_images, dtype=np.float32) / 255.0
    celebA_labels = np.array(celebA_labels)
    DisplayImages(4, celebA_images)
    data_loader = get_data_loader(celebA_images, celebA_labels, batch_size)
else:
    LoadRaf()
    raf_images = np.array(raf_images, dtype=np.float32) / 255.0
    raf_labels = np.array(raf_labels)
    DisplayImages(4, raf_images)
    data_loader = get_data_loader(raf_images, raf_labels, batch_size)

# Create the generator
generator = build_generator(z_dim, img_shape, num_raf_labels=len(emotions), num_celeba_labels=len(selected_features))
# Create the discriminator
discriminator = build_discriminator(img_shape, num_raf_labels=len(emotions), num_celeba_labels=len(selected_features))
generator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.compile(optimizer='adam', loss=discriminator_loss)

# Display model summaries
print("Generator Summary:")
generator.summary()
print("\nDiscriminator Summary:")
discriminator.summary()

# Freeze the discriminator's layers for the combined model
discriminator.trainable = False
# Inputs for the combined model
noise_input = layers.Input(shape=(z_dim,))
label_input = layers.Input(shape=(len(emotions) + len(selected_features) + 2,))
# Generated images
generated_images = generator([noise_input, label_input])
# Discriminator's output
discriminator_output = discriminator([generated_images, label_input])
# Combined model
combined_model = Model([noise_input, label_input], discriminator_output)
# Compile the combined model
combined_model.compile(optimizer='adam', loss='binary_crossentropy')

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    for batch_images, batch_labels in data_loader:
        # Number of images in the batch
        batch = batch_images.shape[0]

        # Prepare label vectors based on the flag
        if flag == 1:  # Celeb-A
            label_vectors = [encode_labels(None, feature_vector, flag) for feature_vector in batch_labels]
        else:  # RAF-DB
            label_vectors = [encode_labels(label, np.zeros(len(selected_features)), flag) for label in batch_labels]

        # Generate noise
        noise = np.random.normal(0, 1, (batch, z_dim))

        # Generate fake images
        generated_images = generator.predict([noise, label_vectors])

        # Labels for real and fake images
        valid = np.ones((batch, 1))
        fake = np.zeros((batch, 1))

        # Train the discriminator
        # Train on real images
        d_loss_real = discriminator.train_on_batch([batch_images, label_vectors], valid)
        # Train on fake images
        d_loss_fake = discriminator.train_on_batch([generated_images, label_vectors], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        g_loss = combined_model.train_on_batch([noise, label_vectors], valid)

        # Print progress
        print(f'Batch Loss - D loss: {d_loss} - G loss: {g_loss}')