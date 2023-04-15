import tensorflow as tf
import os
import glob

def load_preprocessed_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [64, 64])
    image = (image - 127.5) / 127.5
    return image

def create_preprocessed_dataset(image_dir):
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_preprocessed_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

preprocessed_image_dir = "/Users/codylockyear/Desktop/Glover Projects/images"
dataset = create_preprocessed_dataset(preprocessed_image_dir)