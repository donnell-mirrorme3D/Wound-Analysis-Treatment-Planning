import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def wound_segmentation(image_file):
    # Load the image
    image = cv2.imread(image_file)

    # Resize the image to reduce computation time
    image = cv2.resize(image, (400, 400))

    # Convert the image to the L*a*b* color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Guess the skin color
    skin_color = guess_skin_color(lab)

    # Allow the user to refine the skin color selection
    while True:
        # Create a binary mask where the skin pixels are set to 255
        mask = create_skin_mask(lab, skin_color)

        # Display the original and masked images side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Masked")
        plt.show()
        
    # Ask the user if the skin color selection is correct
        answer = input("Is the skin color selection correct? (y/n): ")

        if answer.lower() == 'y':
            break
        else:
            # Allow the user to define the skin color using a polygon tool
            skin_color = define_skin_color(lab)

    # Apply the skin mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Perform wound segmentation
    wound_mask = segment_wound(masked_image)

    # Apply the wound mask to the original image
    masked_image_wound = cv2.bitwise_and(image, image, mask=wound_mask)

    # Display the original and masked images side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Masked")
    axs[2].imshow(cv2.cvtColor(masked_image_wound, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Wound segmentation")
    plt.show()

def guess_skin_color(lab):
    # Get the a* and b* channels of the L*a*b* image
    ab = lab[:, :, 1:].reshape((-1, 2))

    # Apply k-means clustering with 2 clusters (skin and non-skin)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(ab)

    # Assign each pixel to its corresponding cluster
    labels = kmeans.labels_.reshape(lab.shape[:2])

    # Determine the cluster with the smallest average a* and b* values (skin)
    avg_ab_values = [np.mean(ab[labels == i], axis=0) for i in range(2)]
    skin_label = np.argmin([np.linalg.norm(avg_ab_values[i]) for i in range(2)])

    # Get the average L* value (brightness) of the skin pixels
    avg_l_value = np.mean(lab[labels == skin_label, 0])

    # Return the skin color as a tuple of (L*,

    # Calculate the standard deviation of the L* values of the skin pixels
    std_l_value = np.std(lab[labels == skin_label, 0])

    # Define the skin color as a tuple of (L*, a*, b*) values with a standard deviation threshold
    skin_color = (avg_l_value, avg_ab_values[skin_label][0], avg_ab_values[skin_label][1])
    skin_color = refine_skin_color(lab, skin_color, std_l_value)

    return skin_color

def create_skin_mask(lab, skin_color):
    # Create a binary mask where the skin pixels are set to 255
    mask = np.zeros(lab.shape[:2], dtype=np.uint8)
    mask[(lab[:, :, 0] >= skin_color[0] - 20) &
         (lab[:, :, 0] <= skin_color[0] + 20) &
         (lab[:, :, 1] >= skin_color[1] - 20) &
         (lab[:, :, 1] <= skin_color[1] + 20) &
         (lab[:, :, 2] >= skin_color[2] - 20) &
         (lab[:, :, 2] <= skin_color[2] + 20)] = 255

    # Perform morphological closing to fill small holes in the skin region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def refine_skin_color(lab, skin_color, std_l_value):
    # Create a binary mask where the skin pixels are set to 255
    mask = create_skin_mask(lab, skin_color)

    # Get the L* values of the skin pixels
    l_values = lab[mask == 255, 0]

    # Refine the skin color based on the L* value standard deviation
    if np.std(l_values) > std_l_value:
        skin_color = (np.mean(l_values), skin_color[1], skin_color[2])

    return skin_color

def define_skin_color(lab):
    # Display the L*a*b* image
    plt.imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    plt.title("Define skin color using polygon tool")
    plt.show()

    # Allow the user to define the skin color using a polygon tool
    points = plt.ginput(n=-1, show_clicks=True)

    # Create a binary mask where the skin pixels are set to 255 inside the polygon
    mask = np.zeros(lab.shape[:2], dtype=np.uint8)
    points = np.array(points, np.int32)
    cv2.fillPoly(mask, [points], 255)

    # Get the average L*a*b* values of the skin pixels
    skin_color = np.mean(lab[mask == 255], axis=0)

    # Return the skin color as a tuple of (L*, a*, b*) values
    return tuple(skin_color)

def segment_wound(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Perform morphological opening to remove small objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask image for the wound region
    mask = np.zeros_like(mask)

    # Find the contour with the largest area (the wound region)
    max_contour = max(contours, key=cv2.contourArea)

    # Draw the contour on the mask image
    cv2.drawContours(mask, [max_contour], 0, 255, -1)

    return mask

# Define the main function to loop over all images
def main(image_folder_path):
    # Create a list of image file paths
    image_files = [os.path.join(image_folder_path, f) for f in os.listdir(
        image_folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]

    # Loop over each image
    for image_file in image_files:
        wound_segmentation(image_file)

# Define image folder path
image_folder_path = "C:\\Users\\donne\\OneDrive\\Desktop\\wound-data-1"

# Call the main function
main(image_folder_path)

if __name__ == '__main__':
    # Define image folder path
    image_folder_path = "C:\\Users\\donne\\OneDrive\\Desktop\\wound-data-1"

    # Call the main function
    main(image_folder_path)

def create_skin_mask(lab, skin_color):
    # Create a binary mask where the skin pixels are set to 255
    mask = np.zeros(lab.shape[:2], dtype=np.uint8)
    mask[(lab[:, :, 0] >= skin_color[0] - 20) &
         (lab[:, :, 0] <= skin_color[0] + 20) &
         (lab[:, :, 1] >= skin_color[1] - 20) &
         (lab[:, :, 1] <= skin_color[1] + 20) &
         (lab[:, :, 2] >= skin_color[2] - 20) &
         (lab[:, :, 2] <= skin_color[2] + 20)] = 255

    # Perform morphological closing to fill small holes in the skin region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def refine_skin_color(lab, skin_color, std_l_value):
    # Create a binary mask where the skin pixels are set to 255
    mask = create_skin_mask(lab, skin_color)

    # Get the L* values of the skin pixels
    l_values = lab[mask == 255, 0]

    # Refine the skin color based on the L* value standard deviation
    if np.std(l_values) > std_l_value:
        skin_color = (np.mean(l_values), skin_color[1], skin_color[2])

    return skin_color

def define_skin_color(lab):
    # Display the L*a*b* image
    plt.imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    plt.title("Define skin color using polygon tool")
    plt.show()

    # Allow the user to define the skin color using a polygon tool
    points = plt.ginput(n=-1, show_clicks=True)

    # Create a binary mask where the skin pixels are set to 255 inside the polygon
    mask = np.zeros(lab.shape[:2], dtype=np.uint8)
    points = np.array(points, np.int32)
    cv2.fillPoly(mask, [points], 255)

    # Get the average L*a*b* values of the skin pixels
    skin_color = np.mean(lab[mask == 255], axis=0)

    # Return the skin color as a tuple of (L*, a*, b*) values
    return tuple(skin_color)

def ask_yes_no_question(question):
    while True:
        answer = input(f"{question} (y/n) ").lower()
        if answer == 'y':
            return True
        elif answer == 'n':
            return False
        else:
            print("Invalid input, please enter y or n.")


def mask_outside_skin(image, mask):
    # Convert the image to the L*a*b* color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Define the skin color
    skin_color = define_skin_color(lab)

    # Refine the skin color based on the standard deviation of the L* values
    skin_color = refine_skin_color(lab, skin_color, 10)

    # Create a binary mask where the skin pixels are set to 255
    skin_mask = create_skin_mask(lab, skin_color)

    # Combine the wound mask with the skin mask
    combined_mask = cv2.bitwise_or(mask, skin_mask)

    # Invert the mask
    inverted_mask = cv2.bitwise_not(combined_mask)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=inverted_mask)

    # Display the original and masked images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Masked")
    plt.show()

    return masked_image

def wound_segmentation(image_file):
    # Load the image
    image = cv2.imread(image_file)

    # Resize the image to reduce computation time
    image = cv2.resize(image, (400, 400))

    # Segment the wound region
    mask = segment_wound(image)

    # Ask the user to mask everything outside of the skin borders
    while True:
        answer = ask_yes_no_question("Do you want to mask everything outside of the skin borders?")
        if answer:
            masked_image = mask_outside_skin(image, mask)
            break
        else:
            skin_color = define_skin_color(cv2.cvtColor(image, cv2.COLOR_BGR2Lab))
            mask = create_skin_mask(cv2.cvtColor(image, cv2.COLOR_BGR2Lab), skin_color)
            cv2.imshow('Masked', cv2.bitwise_and(image, image, mask=mask))
            cv2.waitKey(0)

    return masked_image
