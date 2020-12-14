# 导入所需工具包 #https://keras.io/
# 已弃用，用来学习

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import initializers
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os

from utils.image_utils import image_utils


def load_data(datas_path, resize_width, resize_height):
    data = []
    labels = []

    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(list(image_utils.list_images(datas_path)))
    random.seed(42)
    random.shuffle(imagePaths)

    # 遍历读取数据
    for imagePath in imagePaths:
        # 读取图像数据，由于使用神经网络，需要给定成一维
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (resize_width, resize_height)).flatten()
        data.append(image)

        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # scale图像数据
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # 数据集切分
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.25, random_state=42)
    return (trainX, testX, trainY, testY)


def build_model():
    # 网络模型结构：3072-512-256-3
    model = Sequential()
    # kernel_regularizer=regularizers.l2(0.01)
    # keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    # initializers.random_normal
    # #model.add(Dropout(0.8))
    model.add(Dense(512, input_shape=(3072,), activation="relu",
                    kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(
        Dense(256, activation="relu", kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(len(lb.classes_), activation="softmax",
                    kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(0.01)))
    return model


def train_model(trainX, trainY, testX, testY, EPOCHS, BATCH_SIZE):
    opt = SGD(lr=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    # 训练网络模型
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                  epochs=EPOCHS, batch_size=BATCH_SIZE)
    return H


def evaluate_model(H, BATCH_SIZE, lable_bin, plot_path):
    predictions = model.predict(testX, batch_size=BATCH_SIZE)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=lable_bin.classes_))

    # 当训练完成时，绘制结果曲线
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N[50:], H.history["accuracy"][50:], label="train_accuracy")
    plt.plot(N[50:], H.history["val_accuracy"][50:], label="val_accuracy")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(plot_path)


if __name__ == '__main__':
    # 设置超参数
    datas_path = r'./data/exp_image'
    model_path = r'./save_models/models/base_train_new_model.h5'
    model_details_path = r'./save_models/details/base_train_new_details.joblib'
    plot_path = r'./train_plot/base_train_new_train_plot.png'
    resize_width = 32
    resize_height = 32

    # 学习率和轮数
    INIT_LR = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32

    print("[INFO] 开始读取数据")
    (trainX, testX, trainY, testY) = load_data(datas_path, resize_width, resize_height)

    # 转换标签，one-hot格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    print("[INFO] 建立模型")
    model = build_model()
    print("[INFO] 准备训练网络...")
    H = train_model(trainX, trainY, testX, testY, EPOCHS, BATCH_SIZE)
    print("[INFO] 正在评估模型")
    evaluate_model(H, BATCH_SIZE, lb, plot_path)

    # 保存模型到本地
    print("[INFO] 正在保存模型")
    model.save(model_path)
    with open(model_details_path, 'wb') as f:
        f.write(pickle.dumps(lb))
