import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from generator import build_generator

# Load the trained generator
latent_dim = 100
generator = build_generator(latent_dim)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Function to generate and save images
def generate_images(model, num_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Generate random noise as input
    noise = tf.random.normal([num_images, latent_dim])

    # Generate images
    generated_images = model(noise, training=False)

    # Save and display images
    for i in range(num_images):
        plt.figure()
        plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'image_{i + 1}.png'))
        plt.show()

num_images = 10
output_dir = 'generated_images'

generate_images(generator, num_images, output_dir)
