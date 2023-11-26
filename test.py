import tensorflow as tf

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from keras.optimizers.legacy import Adam
from visualization_utils import debug_model_prediction

iris_data = load_iris()

x = iris_data['data']
y = iris_data['target']

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_dim=x.shape[1]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, batch_size=10, epochs=3, validation_split=0.2)

model.save('iris.h5')
new_model = tf.keras.models.load_model('iris.h5')

test_loss, test_acc = new_model.evaluate(x,  y, verbose=2)
# test_loss, test_acc = model.evaluate(x,  y, verbose=2)
