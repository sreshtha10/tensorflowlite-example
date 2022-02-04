import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import lite
from tensorflow import keras

dataset_url = 'https://raw.githubusercontent.com/sreshtha10/ML/main/01_Regression/01_Simple%20Linear%20Regression/Salary_Data.csv'
dataset = pd.read_csv(dataset_url)
print(dataset)


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

X = train_dataset.iloc[:,0].values # experience
Y = train_dataset.iloc[:,1].values # salary'

model = keras.Sequential()
model.add(keras.layers.Dense(1,input_shape=[1,]))


model.compile(
  optimizer="sgd", 
  loss="mean_squared_error"
)


model.fit(X,Y,epochs=500)
print(model.predict([10]))


keras_file = "linear.h5"
tf.keras.models.save_model(
    model,
    keras_file
)

converter = lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open("linear.tflite","wb").write(tfmodel)
