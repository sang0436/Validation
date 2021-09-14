import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # cifar-10 dataset 가져오기

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Train samples : ", x_train.shape, y_train.shape)
print("Validation samples : ", x_val.shape, y_val.shape)
print("Test samples : ", x_test.shape, y_test.shape)

print("before normalization")
print("mean : ", np.mean(x_train))
print("std : ", np.std(x_train))

# 정규화 작업
mean = [0, 0, 0]
std = [0, 0, 0]
new_x_train = np.ones(x_train.shape)
new_x_val = np.ones(x_val.shape)
new_x_test = np.ones(x_test.shape)

for i in range(3):
    mean[i] = np.mean(x_train[:, :, :, i])
    std[i] = np.std(x_train[:, :, :, i])

for i in range(3):
    new_x_train[:, :, :, i] = x_train[:, :, :, i] - mean[i]
    new_x_train[:, :, :, i] = new_x_train[:, :, :, i] / std[i]
    new_x_val[:, :, :, i] = x_val[:, :, :, i] - mean[i]
    new_x_val[:, :, :, i] = new_x_val[:, :, :, i] / std[i]
    new_x_test[:, :, :, i] = x_test[:, :, :, i] - mean[i]
    new_x_test[:, :, :, i] = new_x_test[:, :, :, i] / std[i]

x_train = new_x_train
x_val = new_x_val
x_test = new_x_test

print("after normalization")
print("mean : ", np.mean(x_train))
print("std : ", np.std(x_train))

# 일반 모델
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

epochs = 10

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy : ", test_acc)  # Test accuracy :  0.7122

# 가중치 감쇠 적용
model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=(32, 32, 3)))
model2.add(layers.MaxPool2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model2.add(layers.MaxPool2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model2.add(layers.Flatten())
model2.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model2.add(layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.001), activation='softmax'))

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs)
test_loss2, test_acc2 = model2.evaluate(x_test, y_test)
print("Test accuracy (Weight Decay) : ", test_acc2)  # Test accuracy (Weight Decay) :  0.74

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

acc2 = history2.history['accuracy']
val_acc2 = history2.history['val_accuracy']
loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='red', linestyle='solid')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='red', linestyle='dashed')
plt.plot(epochs_range, acc2, label='Training Accuracy (Weight Decay)', color='blue', linestyle='solid')
plt.plot(epochs_range, val_acc2, label='Validation Accuracy (Weight Decay)', color='blue', linestyle='dashed')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='red', linestyle='solid')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='red', linestyle='dashed')
plt.plot(epochs_range, loss2, label='Training Loss (Weight Decay)', color='blue', linestyle='solid')
plt.plot(epochs_range, val_loss2, label='Validation Loss (Weight Decay)', color='blue', linestyle='dashed')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
