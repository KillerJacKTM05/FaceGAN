import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Reshape, Multiply, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D

data_dir = "./RAF-DB"
save_dir = './saved_models/'
img_size = (100, 100)
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_to_label = {emotion: i for i, emotion in enumerate(emotions)}

images = []
labels = []

# Load images and labels
for i, emotion in enumerate(emotions):
    emotion_dir = os.path.join(data_dir, f"{i}_{emotion}")
    for img_file in os.listdir(emotion_dir):
        img_path = os.path.join(emotion_dir, img_file)
        img = Image.open(img_path).resize(img_size)
        images.append(np.array(img))
        labels.append(emotion_to_label[emotion])

images = np.array(images, dtype=np.float32) / 255.0  # Normalize the images to [0, 1]
labels = np.array(labels)

def residual_block(x, filters, kernel_size=3):
    y = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    y = Conv2D(filters, kernel_size=kernel_size, padding='same')(y)
    y = BatchNormalization()(y)
    return layers.add([x, y])

def build_generator(z_dim, num_classes=7):
    input_noise = layers.Input(shape=(z_dim,))
    input_label = layers.Input(shape=(1,), dtype='int32')
    
    # Embedding for categorical input
    label_embedding = layers.Embedding(num_classes, z_dim, input_length=1)(input_label)
    label_embedding = layers.Flatten()(label_embedding)
    
    # Combine noise and label
    x = layers.multiply([input_noise, label_embedding])
    
    x = layers.Dense(128 * 25 * 25)(x)
    x = layers.Reshape((25, 25, 128))(x)
    
    # Use your residual blocks
    x = residual_block(x, 128)
    x = layers.UpSampling2D()(x)
    
    x = Conv2D(64, kernel_size=1, padding='same')(x)  # Adjusting number of channels to 64
    x = residual_block(x, 64)
    x = layers.UpSampling2D()(x)
    
    x = Conv2D(32, kernel_size=1, padding='same')(x)  # Adjusting number of channels to 32
    x = residual_block(x, 32)
    x = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)
    
    model = Model([input_noise, input_label], x)
    return model

def build_discriminator(img_shape, num_classes = 7):
    input_img = layers.Input(shape=img_shape)
    input_label = layers.Input(shape=(1,), dtype='int32')
    
    # Embedding for categorical input
    label_embedding = layers.Embedding(num_classes, np.prod(img_shape), input_length=1)(input_label)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape(img_shape)(label_embedding)
    
    # Combine image and label
    x = layers.Concatenate(axis=-1)([input_img, label_embedding])
    
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model([input_img, input_label], x)
    return model

# Create GAN models
img_shape = (100, 100, 3)
z_dim = 100
# Define loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator = build_generator(z_dim, len(emotions))
discriminator = build_discriminator(img_shape, len(emotions))
# See parameters
generator.summary()
discriminator.summary()

# Randomly select an index
index = np.random.randint(len(images))
# Retrieve the sample image and label
sample_image = images[index]
sample_label = labels[index]
# Use the emotions list to get the emotion name from the label
emotion_name = emotions[sample_label]
plt.imshow(sample_image)
plt.title(emotion_name)
plt.axis('off')
plt.show()

# Get training input
epochs = int(input("Enter the number of epochs: "))
batch_size = int(input("Enter the batch size: "))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Training step function for the discriminator
@tf.function
def train_discriminator(images, labels):
    current_batch_size = len(image_batch)  # Get the current batch size
    noise = np.random.normal(0, 1, (current_batch_size, z_dim))
    
    with tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)
        
        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)
        
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        
    gradients = disc_tape.gradient(total_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

# Training step function for the generator
@tf.function
def train_generator(labels):
    current_batch_size = len(labels)  # Get the current batch size
    noise = np.random.normal(0, 1, (current_batch_size, z_dim))
    
    with tf.GradientTape() as gen_tape:
        generated_images = generator([noise, labels], training=True)
        
        fake_output = discriminator([generated_images, labels], training=True)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        
    gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    for i in tqdm(range(0, len(images), batch_size)):
        image_batch = images[i:i+batch_size]
        label_batch = labels[i:i+batch_size]
        
        train_discriminator(image_batch, label_batch)
        train_generator(label_batch)
        
    # Generate images after the epoch
    noise = tf.random.normal([len(emotions), z_dim])
    labels_tensor = tf.convert_to_tensor(list(range(len(emotions))))
    generated_images = generator([noise, labels_tensor])
    
    fig, axs = plt.subplots(1, len(emotions), figsize=(15, 15))
    for i, img in enumerate(generated_images):
        axs[i].imshow((img + 1) / 2)  # Convert from [-1, 1] to [0, 1]
        axs[i].title.set_text(emotions[i])
        axs[i].axis('off')
    plt.show()
    generator.save_weights(os.path.join(save_dir, f'generator_epoch_{epoch+1}.h5'))