import tensorflow as tf
import os

def convert_to_tf_lite(model):

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  model_path = os.path.dirname(os.path.realpath(__file__)) + '/saved_lite_model/' + 'lite_model';
  # Save the TFLITE model
  with open(f'{model_path}.tflite', 'wb') as f:
      f.write(tflite_model)