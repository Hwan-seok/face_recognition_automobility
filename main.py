import tensorflow as tf
import numpy as np

from datetime import datetime

import os

import matplotlib.pyplot as plt


from tensorflow import keras
from keras.applications import ResNet101
from keras.models import  Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers.legacy import Adam
from keras.preprocessing.image import ImageDataGenerator

from visualization_utils import debug_augmented_images
from visualization_utils import visualize_model_fit
from visualization_utils import debug_model_prediction
from lite_util import convert_to_tf_lite



num_of_classes = 4
image_size = 224
batch_size = 32
epochs = 5
train_directory = os.environ.get("TRAIN_DIRECTORY", "/YOU_MUST_SPECIFY_PATH")

# Load the ResNet50 base model, pre-trained on ImageNet
base_model = ResNet101(weights='imagenet', include_top=False)

# Freeze the layers of the base_model
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_of_classes, activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 
)

train_generator = data_gen.flow_from_directory(
    train_directory,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set as training data
)

validation_generator = data_gen.flow_from_directory(
    train_directory,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# debug_augmented_images(train_generator)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

model_path = f'saved_model/model_{datetime.now()}.h5'
model.save(model_path)
convert_to_tf_lite(model_path)

visualize_model_fit(history)
model = tf.keras.models.load_model('saved_model/main.h5')

debug_model_prediction(validation_generator, model)
