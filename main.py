import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Assuming you have your dataset in 'train_directory'
train_directory = '/Users/lafity101/Downloads/output'

# Initialize the ImageDataGenerator with the rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255, # Rescale the image by normalizing it.
    validation_split=0.2 # if you need a validation split
)

# The size to which you want to resize your images
target_size = (150, 150)

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=target_size, # Resize images to target size
    batch_size=32,
    class_mode='categorical',
    subset='training' # if using validation split
)

validation_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation' # if using validation split
)

# Simple CNN structure
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(4, activation='softmax') # 4 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=5 # You can decide the number of epochs
)

# Save the model if needed
# model.save('path_to_my_model.h5')

