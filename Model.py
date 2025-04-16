import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Path to your CSV dataset file
csv_file = 'Dataset.csv'  # Replace with your CSV file path

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define a function to load and preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    # Read the image file using Keras' load_img (which uses PIL internally)
    img = load_img(image_path, target_size=target_size)
    # Convert image to numpy array
    img_array = img_to_array(img)
    # Rescale the image to [0, 1]
    img_array = img_array / 255.0
    return img_array

# Create a data generator for training with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Create a data generator for validation (no augmentation, just rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Custom generator to load data from DataFrame
def generate_data_from_df(df, batch_size=32):
    while True:
        # Shuffle the DataFrame
        df = df.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(df), batch_size):
            # Get the batch of images and labels
            batch = df.iloc[i:i+batch_size]
            images = []
            labels = []
            for _, row in batch.iterrows():
                image_path = row['image_path']
                label = row['label']
                image = preprocess_image(image_path)
                images.append(image)
                labels.append(label)
            yield np.array(images), np.array(labels)

# Train and validation generators
train_generator = generate_data_from_df(train_df, batch_size=32)
val_generator = generate_data_from_df(val_df, batch_size=32)

# Load VGG16 model without the top (fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification (real or fake)
])

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Callbacks to stop early if validation loss doesn't improve and save the best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // 32,
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_df) // 32,
    callbacks=callbacks
)

# Fine-Tuning (Optional)
# Unfreeze the top 4 layers of the VGG16 base model for fine-tuning
base_model.trainable = True
fine_tune_at = 15
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile the model
model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training the model with fine-tuning
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // 32,
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_df) // 32,
    callbacks=callbacks
)

# Load the best model (from ModelCheckpoint)
model.load_weights('best_model.h5')

# Evaluate the model on the validation data
val_loss, val_acc = model.evaluate(val_generator, steps=len(val_df) // 32)
print(f"Validation Accuracy: {val_acc}")

# Save the final model
model.save('Model.h5')