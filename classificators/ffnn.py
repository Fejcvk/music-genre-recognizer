from tensorflow import keras
from data_processing.data_processor import get_dataset
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import numpy as np
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

dataset1K = get_dataset(sample_len=30)
dataset10K = get_dataset(sample_len=3.0)
dataset = dataset1K
print(dataset.head)
print(dataset.shape)

# Encode the labels
Y = pd.get_dummies(dataset.label).to_numpy()
print(Y[10])

# Scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(np.array(dataset.iloc[:, :-1], dtype=float))
print(X[10])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

print(X_train.shape)
print(Y_train.shape)

batch_count = 50
batch_size = int(X_train.shape[0] / batch_count)

# Simple FFNN
# 3 hidden layers with relu activation + output layer with softmax activation
ffnn = keras.models.Sequential()
ffnn.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
ffnn.add(layers.Dense(128, activation='relu'))
ffnn.add(layers.Dense(64, activation='relu'))
ffnn.add(layers.Dense(10, activation='softmax'))

ffnn.compile(optimizer="adam",
             loss='categorical_crossentropy',
             metrics=['accuracy'])

history = ffnn.fit(X_train, Y_train, epochs=30, batch_size=batch_size, validation_data=(X_test, Y_test))

test_loss, test_acc = ffnn.evaluate(X_test, Y_test)

plt.title("Train accuracy for 1K data")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.xticks(np.arange(0, 30, 5))
plt.yticks(np.arange(0, max(history.history['accuracy']), 0.05))
plt.plot(history.history['accuracy'])
plt.show()

plt.title("Train loss for 1K data")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(history.history['loss'])
plt.show()

plt.title("Validation Accuracy for 1K data")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(history.history['val_accuracy'])
plt.show()

plt.title("Validation loss for 1K data")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(history.history['val_loss'])
plt.show()

print(f"Accuracy on test dataset = {test_acc}")
