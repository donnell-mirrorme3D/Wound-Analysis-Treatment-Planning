import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans

def preprocess(raw):
    image = cv2.resize(raw, (900, 600), interpolation = cv2.INTER_AREA)                                          
    image = image.reshape(image.shape[0]*image.shape[1], 3)
    return image

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        hex_color += ("{:02x}".format(int(i)))
    return hex_color

# Define image folder path
image_folder_path = r"C:\Users\donne\OneDrive\Desktop\Medetec Medical Images"

# Create a list of image file paths
image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# Loop over each image
for image_file in image_files:
    # Load the image
    image = cv2.imread(image_file)

    # Create a copy of the image to draw on
    drawing = image.copy()

    # Create a window to display the image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", (800, 800))
    cv2.imshow("Image", drawing)

    # Create a copy of the original image for highlighting the drawing
    highlight = image.copy()

    # Set up the mouse callback function for drawing
    pts = []
    drawing_done = False
    def draw(event, x, y, flags, param):
        global pts, drawing_done
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            pts.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_done = True

    cv2.setMouseCallback("Image", draw)

    # Wait for the user to finish drawing
    while not drawing_done:
        # Draw the current polygon on the highlight image
        if len(pts) > 1:
            cv2.polylines(highlight, [np.array(pts)], True, (0, 255, 255), thickness=5)

        # Merge the original and highlighted images
        output = cv2.addWeighted(image, 0.7, highlight, 0.3, 0)

        # Display the image with the drawing
        cv2.imshow("Image", output)

# Calculate and display color information
if len(pts) > 2:
    # Create a mask image for the wound region
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Draw the final polygon on the mask
    cv2.fillPoly(mask, [np.array(pts)], 255)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Analyze the color of the masked image
    preprocessed_image = preprocess(masked_image)
    clf = KMeans(n_clusters = 5)
    color_labels = clf.fit_predict(preprocessed_image)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]

    # Analyze colors in the masked image
    color_data = analyze(masked_image)

    # Display the original and masked images with the color information
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Highlight")
    axs[2].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Masked")
    axs[3].pie(color_data["counts"].values(), labels=color_data["hex_colors"], colors=color_data["hex_colors"])
    axs[3].set_title("Color Information")
    plt.show()
