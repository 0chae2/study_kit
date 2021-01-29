import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

# print(tf.__version__)

# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()
#================================================================================
#01 모델 불러오기
df_train = pd.read_csv(r'/home/ch/1.study/0115_deeplearning/ws/train.csv',engine='python',encoding='cp949',header=None)
x_train = df_train.iloc[:,1:].values
y_train = df_train[0].values

df_test = pd.read_csv(r'/home/ch/1.study/0115_deeplearning/ws/t10k.csv',engine='python',encoding='cp949',header=None)
x_test = df_test.iloc[:,1:].values
y_test = df_test[0].values

# img = x_train[1]
# label = y_train[1]
# print(label)
#
# print(img.shape)
# img = img.reshape(28,28)
# print(img.shape)
#
# img_show(img)

print('shape of x_train:', x_train.shape)
print('shape of y_train:', y_train.shape)
print('shape of x_test:', x_test.shape)
print('shape of y_test:', y_test.shape)

#================================================================================
#02 데이터 전처리(정규화, 원핫 인코)딩

#학습 데이터 / 테스트 데이터 정규화 (Nomalization)
x_train = x_train/255
x_test = x_test/255

#정답 데이터 원핫 인코딩 (One-Hot Encoding)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

#================================================================================
#03 모델 구축 및 컴파
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


#================================================================================
#04 모델 학습
print('\n\n\n\n')
print('Start learning')
hist = model.fit(x_train, y_train, epochs=30, validation_split=0.3)


#================================================================================
#05 모델(정확) 평가
print('\n\n\n\n\n')
print('###Evaluation###')
model.evaluate(x_test, y_test)


#================================================================================
#06 손실 및 정확도

plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')

plt.legend(loc='best')

# plt.show()

plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')

plt.legend(loc='best')

plt.show()

plt.figure(figsize=(6,6))

predicated_value = model.predict(x_test)

cm = confusion_matrix(np.argmax(y_test, axis=1),
                      np.argmax(predicated_value, axis=-1))

sns.heatmap(cm, annot=True, fmt='d')
plt.show()

print(cm)
print('\n')

for i in range(10):
    print(('label = %d\t (%d/%d)\t accuracy = %.3f') %
          (i, np.max(cm[i]), np.sum(cm[i]),
           np.max(cm[i])/np.sum(cm[i])))

print('\n\nFinish & save model')
model.save('mnist_model_01.h5')
