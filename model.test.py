import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import numpy as np
from PIL import Image

# Define the model architecture
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# Load the weights from the saved model file
model.load_weights('./transferepochoverfit.keras')

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Function to preprocess the input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to match the model's input size
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Path to the input image
# dataset\testing_data\cracked\Cracked-7.jpg
image_path = '../dataset/testing_data/cracked/Cracked-1.jpg'

# Preprocess the input image
input_image = preprocess_image(image_path)

# Make predictions using the loaded model
predictions = model.predict(input_image)
print(predictions[0])

# Assuming binary classification (crack vs. no crack)
if predictions[0] > 0.5:
    print("Tire crack detected!")
else:
    print("No tire crack detected.")