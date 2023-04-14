ARTistGen

ARTistGen is a deep learning project that generates artistic images using a DCGAN (Deep Convolutional Generative Adversarial Network) architecture. This project utilizes TensorFlow for training the generator and discriminator models, and aims to generate unique art styles based on a custom dataset of preprocessed images. With ARTistGen, you can experience the power of AI-generated art and enjoy the endless possibilities of creativity driven by cutting-edge machine learning technology.

Installation

Clone the repository:
bash
Copy code
git clone https://github.com/your_username/ARTistGen.git
Install the required packages:
Copy code
pip install -r requirements.txt
Usage

Preprocess your dataset of images using the preprocess.py script. Make sure to update the paths for the input and output directories within the script.
Copy code
python preprocess.py
Train the DCGAN model with your preprocessed dataset using the dcgan_train.py script.
Copy code
python dcgan_train.py
During the training process, images will be generated and saved to visualize the progress. You can view these images to see the quality of the generated art.
After training is complete, you can use the trained model to generate new images based on the learned art styles.
Contributing

Feel free to open issues or submit pull requests for improvements, bug fixes, or new features. We appreciate any feedback and contributions to make ARTistGen better.

License

This project is licensed under the MIT License. See the LICENSE file for more details.
