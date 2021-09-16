import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random
import cv2

# cifar-10 dataset 가져오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 이미지 시각화
for i in range(1, 11):
    ax = plt.subplot(2, 5, i)
    ax.imshow(x_train[i])
    ax.set_title(np.array(class_names)[y_train[i]])
plt.show()

# Validation set 만들기
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=1)

# sample 정보
print("Train samples : ", x_train.shape, y_train.shape)
print("Validation samples : ", x_val.shape, y_val.shape)
print("Test samples : ", x_test.shape, y_test.shape)

# 정규화 전 평균,표준편차
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

# 정규화 후 평균, 표준편차
print("after normalization")
print("mean : ", np.mean(x_train))
print("std : ", np.std(x_train))

epochs = 20

# 훈련 모델
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.001), activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, validation_data=(x_val, y_val), epochs=epochs, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy : ", test_acc)  # Test accuracy : 0.7518

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

# 훈련 과정 시각화 (정확도)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue', linestyle='solid')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='blue', linestyle='dashed')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

# 훈련 과정 시각화 (손실)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='red', linestyle='solid')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='red', linestyle='dashed')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# test
i = random.randint(0, 10000 - 1)
plt.imshow(np.array(x_test[i]))
resized_img = cv2.resize(x_test[i], (32, 32))
resized_img = np.expand_dims(resized_img, axis=0)
pre = model.predict(resized_img)
n = np.argmax(pre)
print("Output Label", n)
print(class_names[n])
plt.show()
