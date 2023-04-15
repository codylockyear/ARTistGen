import pandas as pd
import os
import cv2

def process_image(image_path, output_directory):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (256, 256))
    preprocessed_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(preprocessed_path, resized_image)

import pandas as pd
import os
import cv2

def process_image(image_path, output_directory):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (256, 256))
    preprocessed_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(preprocessed_path, resized_image)

import pandas as pd
import os
import cv2

def process_image(image_path, output_directory):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (256, 256))
    preprocessed_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(preprocessed_path, resized_image)

if __name__ == "__main__":
    # Update these paths as needed
    dataset_directory = "/Users/codylockyear/Desktop/Glover Projects"
    train_info_csv_path = os.path.join(dataset_directory, "train_info.csv")
    train_info_df = pd.read_csv(train_info_csv_path)
    
    # Limit the number of processed images
    train_info_df = train_info_df.head(100)

    output_directory = "images"
    os.makedirs(output_directory, exist_ok=True)
    for index, row in train_info_df.iterrows():
        artist_name = row["artist"]
        file_name = row["filename"]
        image_path = os.path.join(dataset_directory, "train", file_name)

        print(f"Processing {image_path}")
        process_image(image_path, output_directory)


