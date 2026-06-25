import cv2
import numpy as np
# Load the image in grayscale
img = cv2.imread("./processed_dataset_split/test/triangle/triangle_1212.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found or unable to load.")
    exit()

# Apply binary thresholding to convert the image to a binary image
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(
    binary, 
    cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

# Select the largest contour based on area
contour = max(contours, key=cv2.contourArea)

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
    vertices]

print("Feature Vector:", feature_vector)
