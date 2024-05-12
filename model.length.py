from tensorflow.keras.applications import MobileNetV2

# Load MobileNetV2 model with ImageNet weights
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Get the total number of layers in the model
num_layers = len(model.layers)
print("Total number of layers in MobileNetV2:", num_layers)
