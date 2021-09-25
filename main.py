import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

# cifar-10 dataset 가져오기
from tensorflow_core.python.keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 이미지 시각화
for i in range(1, 11):
    ax = plt.subplot(2, 5, i)
    ax.imshow(x_train[i])
    ax.set_title(np.array(class_names)[y_train[i]])
plt.show()

# sample 정보
print("Train samples : ", x_train.shape, y_train.shape)
print("Test samples : ", x_test.shape, y_test.shape)

# 정규화 및 데이터 늘리기
train_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                             width_shift_range=0.125, height_shift_range=0.125,
                             horizontal_flip=True)
test_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
for data in (train_gen, test_gen):
    data.fit(x_train)

# 원 핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

epochs = 100

# 훈련 모델
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu',
                        input_shape=(32, 32, 3)))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.001), activation='softmax'))

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen.flow(x_train, y_train, batch_size=32),
                    epochs=epochs, verbose=2, validation_data=test_gen.flow(x_test, y_test, batch_size=32),
                    validation_steps=x_test.shape[0]//128)
test_loss, test_acc = model.evaluate(test_gen.flow(x_test, y_test, batch_size=32))
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
plt.rcParams["figure.figsize"] = (2, 2)
for i in range(10):
    output = model.predict(x_test[i].reshape(0, 32, 32, 3))
    plt.imshow(x_test[i].reshape(32, 32, 3))
    print("예측 : " + class_names[np.argmax(output)] + "/ 정답 : " + class_names[np.argmax(y_test[i])])
    plt.show()
