import matplotlib

matplotlib.use("Agg")

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
import cv2
from model import Autoencoder

TRAIN_DATA = 'data/train'
MODEL = 'output/model.h5'
PLOT_PATH = 'plot.png'

EPOCHS = 10
BS = 10

IMAGE_HEIGHT = 252
IMAGE_WIDTH = 252

# инициализируем данные и метки
print("[INFO] loading images...")
data = []
image_paths = list(paths.list_images(TRAIN_DATA))

# цикл по изображениям
for image_path in image_paths:
    image = cv2.imread(image_path)

    # убрали третье значение (канал)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    data.append(image)

# разбиваем данные на обучающую (75%) и тестовую выборки (25%)
trainX, testX = train_test_split(data, test_size=0.2)

# добавить 1 вместо канала
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

trainX = np.asarray(trainX).astype("float32") / 255.0
testX = np.asarray(testX).astype("float32") / 255.0

# шумы
trainNoise = np.random.normal(loc=0.5, scale=0.5, size=trainX.shape)
testNoise = np.random.normal(loc=0.5, scale=0.5, size=testX.shape)
trainXNoisy = np.clip(trainX + trainNoise, 0, 1)
testXNoisy = np.clip(testX + testNoise, 0, 1)

print("[INFO] building autoencoder...")
opt = Adam(lr=1e-3)

autoencoder = Autoencoder().build(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
autoencoder.compile(loss="mse", optimizer=opt)

H = autoencoder.fit(
    trainXNoisy, trainX,
    validation_data=(testXNoisy, testX),
    epochs=EPOCHS,
    batch_size=BS)

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)

autoencoder.save(MODEL)
