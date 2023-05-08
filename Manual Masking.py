import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Set the paths to the images folder and the masks folder
images_path = "C:\\Users\\donne\\OneDrive\\Desktop\\wound-data-1\\images"
masks_path = "C:\\Users\\donne\\OneDrive\\Desktop\\wound-data-1\\masks"

# Create the masks folder if it doesn't exist
if not os.path.exists(masks_path):
    os.makedirs(masks_path)

# Load the image filenames
image_filenames = os.listdir(images_path)

# Loop through the images and mask them
for filename in image_filenames:
    # Load the image
    image_path = os.path.join(images_path, filename)
    image = cv2.imread(image_path)

    # Create a blank mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Display the image and allow the user to draw the polygon
    plt.imshow(image)
    polygon_points = plt.ginput(n=-1, timeout=0)
    polygon_points = np.array(polygon_points, dtype=np.int32)

    # Draw the polygon on the mask
    cv2.fillPoly(mask, [polygon_points], 255)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Save the masked image to the masks folder
    mask_filename = filename.split(".")[0] + "_mask.png"
    mask_path = os.path.join(masks_path, mask_filename)
    cv2.imwrite(mask_path, mask)

    # Save the masked image to the images folder
    masked_image_filename = filename.split(".")[0] + "_masked.png"
    masked_image_path = os.path.join(images_path, masked_image_filename)
    cv2.imwrite(masked_image_path, masked_image)

    # Close the plot window
    plt.close()

print("Masking complete.")
