import numpy as np
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models, layers
from tensorflow.keras.layers import BatchNormalization, Activation
import tensorflow as tf
from sklearn.model_selection import train_test_split

# CIFAR-10 dataset 가져오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 이미지 시각화
plt.figure(figsize=[10, 10])
for i in range(25):    # 1~25번째 이미지
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(x_train[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[y_train[i][0]])
plt.show()

# validation set 만들기
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=1)

# sample 정보
print("Train samples : ", x_train.shape, y_train.shape)
print("Validation samples : ", x_val.shape, y_val.shape)
print("Test samples : ", x_test.shape, y_test.shape)

# 데이터 확대
gen = ImageDataGenerator(rotation_range=20, shear_range=0.2,
                         width_shift_range=0.2, height_shift_range=0.2,
                         horizontal_flip=True)
augment_ratio = 1.5  # 40000*1.5 = 60000개의 train sample 추가
augment_size = int(augment_ratio * x_train.shape[0])
randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
x_augmented, y_augmented = gen.flow(x_augmented, y_augmented, batch_size=augment_size,
                                    shuffle=False).next()
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
s = np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]
print("after augmented (train set) : ", x_train.shape, y_train.shape)

# 픽셀값 정규화
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_val = x_val / 255
x_test = x_test / 255

# 원 핫 인코딩
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

epochs = 20

# 훈련 모델
model = models.Sequential()
# Conv Layer1
model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Conv Layer2
model.add(layers.Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Pooling Layer1
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.3))  # Dropout
# Conv Layer3
model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Conv Layer4
model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Pooling Layer2
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.5))  # Dropout
# Conv Layer5
model.add(layers.Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Conv Layer6
model.add(layers.Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Pooling Layer3
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.5))  # Dropout
# Flat Layer
model.add(layers.Flatten())
# Dense Layer1
model.add(layers.Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(layers.Dropout(0.5))  # Dropout
# Dense Layer2
model.add(layers.Dense(10, activation='softmax'))

# 모델 학습
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_data=(x_val, y_val))
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy : ", test_acc)  # Test accuracy : 0.8187

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

# Test
predict = model.predict(x_test)
predict_classes = np.argmax(predict, axis=1)
fig, axes = plt.subplots(5, 5, figsize=(15, 15))
axes = axes.ravel()
for i in np.arange(0, 25):
    axes[i].imshow(x_test[i])
    axes[i].set_title("True: %s \nPredict: %s" % (class_names[np.argmax(y_test[i])], class_names[predict_classes[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)

