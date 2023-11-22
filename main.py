import os

# отключение гпу обработки
# (если используется встроенная графика)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# импорт необходимых библиотек
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten

# инициализация тренировочной
# и тестовой выборок
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# нормализация входных данных
x_train = x_train / 255
x_test = x_test / 255

# преобразование выходных данных
# в вектора длиной 10
y_train_vect = keras.utils.to_categorical(y_train, 10)
y_test_vect = keras.utils.to_categorical(y_test, 10)

# структура сети: полносвязная сеть
# во входной слой подаются изображения 28*28 (784 входа)
# скрытый слой содержит 100 нейронов, функция активации - relu
# выходной слой содержит 10 выходов, функция активации - softmax
# (для использования категориальной кросс-энтропии)
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())

# компиляция сети: оптимизация по adam,
# критерий качества - категориальная кросс-энтропия,
# в качестве метрики - кол-во верно классифицированых объектов
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='accuracy')

# обучение сети на 5 эпохах:
# 80% - обучающая выборка, 20% - валидация
# обновляем веса каждые 32 объекта
model.fit(x_train, y_train_vect, batch_size=32, epochs=5, validation_split=0.2)

# проверка сети
for i in range(5):
    n = random.randint(0, 10000)
    x = np.expand_dims(x_test[n], axis=0)
    res = model.predict(x)
    print(f"Распознанная цифра {np.argmax(res)}")

    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()
