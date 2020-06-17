from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import numpy as np
import pickle
import cv2
import os
import matplotlib.pyplot as plt

PLOT_PATH = 'loss-accuracy.png'
TRAIN_DATA = 'data/train'
MODEL = 'output/model.pickle'
LABEL = 'output/labels.pickle'
EPOCHS = 65

# инициализируем данные и метки
print("[INFO] loading images...")
data = []
labels = []

image_paths = list(paths.list_images(TRAIN_DATA))

# цикл по изображениям
for image_path in image_paths:
    # загружаем изображение, меняем размер на 32x32 пикселей,  сглаживаем его в 32x32x3=3072 пикселей и
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)

    # извлекаем название класса по имени папки, в которой лежит картинка (cats, dogs, panda)
    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

# масштабируем интенсивности пикселей в диапазон [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# разбиваем данные на обучающую (75%) и тестовую выборки (25%)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.10)

# конвертируем метки из целых чисел в векторы
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# определим архитектуру 3072-1024-512-3
# TODO: почему такая архитектура, почему такие функции активации? Какие варианты???
# 3072 - размерность входа, далее свёртки 3, 2
# в качестве функции активации следует использовать ту или иную функцию из класса сигмоид,
# так как она позволяет усиливать слабые сигналы и не насыщаться от сильных.
# Плюс производную сигмоиды можно легко выразить через неё саму, что существенно ускоряет вычисления
# Во внтуреннем слое следует использовать tanh, так как он центрует данные относительно нуля,
# ввиду чего обучение на следующем слое осуществляется заметно лучше
# sigmoid лучше использовать ближе к выходу, чтобы провести сглаживание + на выходе она не так сильно будет убивать градиент
# softmax часто используется в качестве последнего слоя глубоких НС для задач классификации (k > 2)
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="tanh"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# компилируем модель, используя SGD как оптимизатор и категориальную кросс-энтропию в качестве функции потерь
# TODO: какой оптимизатор и функцию потерь можно предложить???
# так как используется softmax, то в качестве функции потерь ддолжна использоваться перекрёстная энтропия
# имеет смысл использовать стохастический градиентный спуск, ввиду того, что каждая итерация вычисляется довольно бытро,
# также он хорош тем, что на каждом шаге есть шанс, что очередное значение "выбьет" из локального минимума
# и итоговое значение будет более оптимальным
print("[INFO] training network...")
INIT_LR = 0.01
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# обучаем нейросеть
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# оцениваем нейросеть
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# строим графики потерь и точности
n_epochs = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(n_epochs, history.history["loss"], label="training loss")
plt.plot(n_epochs, history.history["val_loss"], label="validation loss")
plt.plot(n_epochs, history.history["accuracy"], label="training accuracy")
plt.plot(n_epochs, history.history["val_accuracy"], label="validation accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(PLOT_PATH)

# сохраняем модель и метки на диск
print("[INFO] serializing network and label binarizer...")
model.save(MODEL)
f = open(LABEL, "wb")
f.write(pickle.dumps(lb))
f.close()
