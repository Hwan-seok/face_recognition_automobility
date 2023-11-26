import matplotlib.pyplot as plt
import numpy as np
import os 

dirname = os.path.dirname(os.path.realpath(__file__))

##### Debug augmented images

def debug_augmented_images(train_generator):

  x_batch, _ = next(train_generator)
  x_batch = x_batch * 255
  plt.figure(figsize=(14, 7))
  for i in range(0, 10):
      plt.subplot(2, 5, i+1)
      plt.imshow(x_batch[i].astype('uint8'))
      plt.axis('off')
  plt.tight_layout()
  plt.savefig(f'{dirname}/figures/augmented_images_debug.png')



def debug_model_prediction(validation_generator, model):
  # Get the class indices
  class_indices = validation_generator.class_indices

  # Convert the dictionary of class indices to a list of class names, sorted by their index value
  class_names = list(class_indices.keys())

  (images, labels) = next(validation_generator)

  # Make predictions
  predictions = model.predict(images)
  predicted_classes = np.argmax(predictions, axis=1)
  true_classes = np.argmax(labels, axis=1)
  
  images_display = images * 255
  images_display = images_display.astype(np.uint8)
  
  # Plot the images and the predicted vs true labels
  plt.figure(figsize=(10, 9))
  for i in range(images_display.shape[0]):
      plt.subplot(4, 8, i + 1)
      plt.imshow(images_display[i].astype("uint8"))
      plt.title(f"Pred: {class_names[predicted_classes[i]]}\nTrue: {class_names[true_classes[i]]}")
      plt.axis("off")
  
  plt.tight_layout()
  plt.savefig(f'{dirname}/figures/model_prediction_debug.png')

def visualize_model_fit(history): 
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')

  plt.savefig(f'{dirname}/figures/model_fit.png')