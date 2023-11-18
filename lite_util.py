import tensorflow as tf

def convert_to_tf_lite(path):

  converter = tf.lite.TFLiteConverter.from_saved_model(path)
  tflite_model = converter.convert()

  # Save the TFLITE model
  with open(f'{path}.tflite', 'wb') as f:
      f.write(tflite_model)