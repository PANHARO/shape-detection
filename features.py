import cv2
import os
import numpy as np

X_train, y_train = [], []
TRAIN_PATH = "./processed_dataset_split/train"

for class_name in os.listdir(TRAIN_PATH):
    class_dir = os.path.join(TRAIN_PATH, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        try:
            # Load the image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error: Image not found or unable to load: {img_path}")
                exit()

            # Apply binary thresholding to convert the image to a binary image
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # Find contours in the binary image
            contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)

            if contours is None or len(contours) == 0:
                print(f"No contours found in image: {img_path}")
                continue

            # Select the largest contour based on area
            contour = max(contours, key=cv2.contourArea)

            # Hu Moments
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()

            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            x,y,w,h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h

            circularity = (4 * np.pi * area) / (perimeter ** 2)

            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)

            feature_vector = [
            area, 
            perimeter, 
            aspect_ratio, 
            circularity,
            vertices,
            *hu_moments]

            X_train.append(feature_vector)
            y_train.append(class_name)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
