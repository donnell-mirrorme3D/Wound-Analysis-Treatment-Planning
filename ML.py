# Import the required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Set the path to the dataset
dataset_path = "C:\\Users\\donne\\OneDrive\\Desktop\\wound-data-1"

# Define the input and output image dimensions
input_shape = (256, 256, 3)
output_channels = 1

# Load the segmented and normal images
segmented_images_path = os.path.join(dataset_path, "segmented")
normal_images_path = os.path.join(dataset_path, "normal")
segmented_images = []
normal_images = []

for filename in os.listdir(segmented_images_path):

    # Load the segmented image and resize it to the input shape
    segmented_image = cv2.imread(os.path.join(segmented_images_path, filename))
    segmented_image = cv2.resize(segmented_image, input_shape[:2])

    # Add the segmented image to the list of segmented images
    segmented_images.append(segmented_image)

    # Load the corresponding normal image and resize it to the input shape
    normal_image_filename = filename.split("_mask")[0] + ".jpg"
    normal_image = cv2.imread(os.path.join(normal_images_path, normal_image_filename))
    normal_image = cv2.resize(normal_image, input_shape[:2])

    # Add the normal image to the list of normal images
    normal_images.append(normal_image)

# Convert the lists of images to NumPy arrays
segmented_images = np.array(segmented_images)
normal_images = np.array(normal_images)

# Normalize the pixel values of the images
segmented_images = segmented_images.astype("float32") / 255.0
normal_images = normal_images.astype("float32") / 255.0

# Define the CNN model
inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(512, 3, activation="relu", padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", padding="same")(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
outputs = layers.Conv2D(output_channels, 1, activation="sigmoid", padding="same")(x)

# Define the CNN model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(normal_images, segmented_images, batch_size=32, epochs=10, validation_split=0.2)

# Evaluate the model on a test set
test_loss, test_acc = model.evaluate(test_normal_images, test_segmented_images)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

# Save the trained model
model.save("wound_segmentation_model.h5")

# Load the saved model
loaded_model = keras.models.load_model("wound_segmentation_model.h5")

# Use the loaded model to predict segmentation masks for new images
new_images = np.array([...]) # Load new images here
normalized_images = new_images.astype("float32") / 255.0
predictions = loaded_model.predict(normalized_images)

# Visualize the predicted segmentation masks
for i in range(len(predictions)):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(new_images[i])
    axs[0].set_title("Original Image")
    axs[1].imshow(predictions[i, :, :, 0])
    axs[1].set_title("Segmentation Mask")
    plt.show()

    # Apply post-processing techniques to the predicted segmentation masks
processed_predictions = []
for pred in predictions:
    # Apply thresholding to the predicted mask
    thresholded = cv2.threshold(pred[:, :, 0], 0.5, 1, cv2.THRESH_BINARY)[1]
    
    # Apply morphological operations to the thresholded mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    processed_predictions.append(opening)
    
# Visualize the processed predicted masks
for i in range(len(processed_predictions)):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(new_images[i])
    axs[0].set_title("Original Image")
    axs[1].imshow(processed_predictions[i])
    axs[1].set_title("Segmentation Mask")
    plt.show()

    # Apply the processed predicted masks to the original images
for i in range(len(processed_predictions)):
    masked_image = cv2.bitwise_and(new_images[i], new_images[i], mask=processed_predictions[i])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(new_images[i])
    axs[0].set_title("Original Image")
    axs[1].imshow(masked_image)
    axs[1].set_title("Masked Image")
    plt.show()

# Save the processed predicted masks as image files
for i in range(len(processed_predictions)):
    mask_path = f"mask_{i}.png"
    cv2.imwrite(mask_path, (255*processed_predictions[i]).astype(np.uint8))
# Load the saved processed predicted masks
loaded_masks = []
for i in range(len(processed_predictions)):
    mask_path = f"mask_{i}.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
    loaded_masks.append(mask)
    
# Visualize the loaded masks
for i in range(len(loaded_masks)):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(new_images[i])
    axs[0].set_title("Original Image")
    axs[1].imshow(loaded_masks[i])
    axs[1].set_title("Loaded Mask")
    plt.show()

# Compute metrics to evaluate the segmentation performance
intersection = np.sum(np.logical_and(loaded_masks, true_masks))
union = np.sum(np.logical_or(loaded_masks, true_masks))
iou_score = intersection / union
precision = np.sum(np.logical_and(loaded_masks, true_masks)) / np.sum(loaded_masks)
recall = np.sum(np.logical_and(loaded_masks, true_masks)) / np.sum(true_masks)

# Print the evaluation metrics
print("IoU score:", iou_score)
print("Precision:", precision)
print("Recall:", recall)

# Save the trained model for future use
model.save("wound_segmentation_model.h5")

# Load the saved model
loaded_model = tf.keras.models.load_model("wound_segmentation_model.h5")

# Test the loaded model on new images
new_images = []
for i in range(3):
    image_path = f"new_image_{i}.jpg"
    image = cv2.imread(image_path)
    new_images.append(image)

# Preprocess the new images
processed_images = preprocess_images(new_images)

# Use the loaded model to predict the masks for the new images
loaded_predictions = loaded_model.predict(processed_images)

# Apply the processed predicted masks to the original images
for i in range(len(loaded_predictions)):
    masked_image = cv2.bitwise_and(new_images[i], new_images[i], mask=loaded_predictions[i])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(new_images[i])
    axs[0].set_title("Original Image")
    axs[1].imshow(masked_image)
    axs[1].set_title("Masked Image")
    plt.show()

# Compute the IoU score, precision, and recall for the loaded model predictions
loaded_processed_predictions = process_predictions(loaded_predictions)
loaded_intersection = np.sum(np.logical_and(loaded_processed_predictions, true_masks))
loaded_union = np.sum(np.logical_or(loaded_processed_predictions, true_masks))
loaded_iou_score = loaded_intersection / loaded_union
loaded_precision = np.sum(np.logical_and(loaded_processed_predictions, true_masks)) / np.sum(loaded_processed_predictions)
loaded_recall = np.sum(np.logical_and(loaded_processed_predictions, true_masks)) / np.sum(true_masks)

# Print the evaluation metrics for the loaded model predictions
print("Loaded model IoU score:", loaded_iou_score)
print("Loaded model precision:", loaded_precision)
print("Loaded model recall:", loaded_recall)

# Visualize the predicted masks for the loaded model
for i in range(len(loaded_predictions)):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(new_images[i])
    axs[0].set_title("Original Image")
    axs[1].imshow(loaded_predictions[i])
    axs[1].set_title("Predicted Mask")
    plt.show()

# Train the model using the labeled dataset
model.fit(train_images, train_masks, epochs=num_epochs, validation_data=(val_images, val_masks))

# Evaluate the trained model on the test dataset
test_loss, test_iou_score, test_precision, test_recall = model.evaluate(test_images, test_masks)

# Print the evaluation metrics for the trained model
print("Test loss:", test_loss)
print("Test IoU score:", test_iou_score)
print("Test precision:", test_precision)
print("Test recall:", test_recall)

# Use the trained model to predict masks for the test dataset
predictions = model.predict(test_images)

# Process the predictions by thresholding and converting to binary masks
processed_predictions = process_predictions(predictions)

# Compute the intersection over union (IoU) score, precision, and recall for the trained model predictions
intersection = np.sum(np.logical_and(processed_predictions, test_masks))
union = np.sum(np.logical_or(processed_predictions, test_masks))
iou_score = intersection / union
precision = np.sum(np.logical_and(processed_predictions, test_masks)) / np.sum(processed_predictions)
recall = np.sum(np.logical_and(processed_predictions, test_masks)) / np.sum(test_masks)

# Print the evaluation metrics for the trained model predictions
print("Trained model IoU score:", iou_score)
print("Trained model precision:", precision)
print("Trained model recall:", recall)

# Visualize the predicted masks for the trained model
for i in range(len(predictions)):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(test_images[i])
    axs[0].set_title("Original Image")
    axs[1].imshow(predictions[i])
    axs[1].set_title("Predicted Mask")
    plt.show()

# Save the trained model to disk
model.save("wound_segmentation_model.h5")

# Load the saved model from disk
loaded_model = tf.keras.models.load_model("wound_segmentation_model.h5")

# Use the loaded model to predict masks for new images
new_predictions = loaded_model.predict(new_images)

# Process the new predictions by thresholding and converting to binary masks
processed_new_predictions = process_predictions(new_predictions)

# Visualize the predicted masks for the new images
for i in range(len(new_predictions)):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(new_images[i])
    axs[0].set_title("Original Image")
    axs[1].imshow(processed_new_predictions[i])
    axs[1].set_title("Predicted Mask")
    plt.show()
