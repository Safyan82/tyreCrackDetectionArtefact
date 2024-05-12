import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

# Custom dataset class
class TyreCrackDataset(tf.keras.utils.Sequence):
    def __init__(self, generator, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator

    def __getitem__(self, index):
        x, y = self.generator[index]
        return x, y

    def __len__(self):
        return len(self.generator)

# Set up data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/My Drive/dataset/training_data',
    target_size=(224, 224),
    batch_size=4,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    '/content/drive/My Drive/dataset/testing_data',
    target_size=(224, 224),
    batch_size=4,
    class_mode='binary'
)

# Create instances of the custom dataset
train_dataset = TyreCrackDataset(train_generator)
val_dataset = TyreCrackDataset(val_generator)

# Load pre-trained EfficientNetB3 model
base_model = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=25
)

# Fine-tune the model
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=25
)

# Save the model
model.save('tyremodel.keras')

# Visualize training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_fine_tune.history['accuracy'], label='Training Accuracy (Fine-tuned)')
plt.plot(history_fine_tune.history['val_accuracy'], label='Validation Accuracy (Fine-tuned)')
plt.title('Training and Validation Accuracy (Fine-tuned)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_fine_tune.history['loss'], label='Training Loss')
plt.plot(history_fine_tune.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.tight_layout()
plt.show()

# Visualize model predictions
def visualize_predictions(model, generator, num_samples=5):
    fig, axes = plt.subplots(nrows=1, ncols=num_samples, figsize=(20, 4))

    for i in range(num_samples):
        x, y_true = generator[i]
        y_pred = model.predict(x)
        y_pred_class = np.round(y_pred)

        ax = axes[i]
        ax.imshow(x[0])
        ax.set_title(f"True: {y_true[0]}, Pred: {y_pred_class[0]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

visualize_predictions(model, val_generator)