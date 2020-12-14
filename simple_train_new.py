# 导入所需工具包 #https://keras.io/
import math
import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os
from utils.image_utils import image_utils
from utils.model_utils.model_details import SavedModelDetails
from utils.model_utils.build_net import SimpleVGGNet, SimpleNet


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
        image = cv2.resize(image, (resize_width, resize_height))
        data.append(image)
        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    # scale图像数据
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    # 数据集切分
    (trainValX, testX, trainValY, testY) = train_test_split(data,
                                                            labels, test_size=0.15, random_state=42)
    (trainX, valX, trainY, valY) = train_test_split(trainValX,
                                                    trainValY, test_size=0.15, random_state=42)
    return (trainX, valX, testX, trainY, valY, testY)

def generate_trainxy(trainX, trainY, BATCH_SIZE):
    total_size = trainX.shape[0]
    # 早前使用的方法，没有随机抽取样本
    # count = 1
    # while 1:
    #     batch_x = trainX[(count - 1) * BATCH_SIZE:count * BATCH_SIZE]
    #     batch_y = trainY[(count - 1) * BATCH_SIZE:count * BATCH_SIZE]
    #     count = count + 1
    #     if count * BATCH_SIZE >= total_size:
    #         count = 1
    #     yield (batch_x, batch_y)
    while 1:
        random_nums=[]
        for random_num_i in range(BATCH_SIZE):
            random_num = random.randint(0, total_size - 1)
            random_nums.append(random_num)
        batch_x = trainX[random_nums]
        batch_y = trainY[random_nums]
        yield (batch_x, batch_y)

def train_model(trainX, trainY, validateX, validateY, EPOCHS, BATCH_SIZE, model, INIT_LR, model_path):
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # 有一次提升, 则覆盖一次,保证最好的一次被保存
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1,
                                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # 训练网络,如果不使用数据增强，则使用generate_trainxy(trainX, trainY, BATCH_SIZE)代替aug.flow(...,...,...)
    H = model.fit_generator(generate_trainxy(trainX, trainY, BATCH_SIZE),
                            validation_data=(validateX, validateY), steps_per_epoch=len(trainX) // BATCH_SIZE,
                            epochs=EPOCHS, callbacks=callbacks_list)
    # H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    #                         validation_data=(validateX, validateY), steps_per_epoch=len(trainX) // BATCH_SIZE,
    #                         epochs=EPOCHS, callbacks=callbacks_list)

    # 如果没有checkpoint则不需要callbacks
    # H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    #                         validation_data=(validateX, validateY), steps_per_epoch=len(trainX) // BATCH_SIZE,
    #                         epochs=EPOCHS)

    return H

def evaluate_model(model, EPOCHS, testX, testY, H, BATCH_SIZE, lable_bin, plot_path):
    predictions = model.predict(testX, batch_size=BATCH_SIZE)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=lable_bin.classes_))

    # 当训练完成时，绘制结果曲线
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    xlabel_interval = math.floor(EPOCHS / 50)
    plt.plot(N[xlabel_interval:], H.history["accuracy"][xlabel_interval:], label="train_accuracy")
    plt.plot(N[xlabel_interval:], H.history["val_accuracy"][xlabel_interval:], label="val_accuracy")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(plot_path)

def train_process(image_path, save_model_path, save_model_details_path, save_plot_path, resize_width, resize_height):
    # 设置超参数
    # 用原始图像用cat_dog_image
    # datas_path = r'./data/cat_dog_image'
    # 用混淆过的图像用cat_dog_generate
    image_path = image_path
    # 需要保存模型用这一句，后面savemodel需要放开注释
    # model_path = r'./save_models/models/vgg_train_cat_dog_model.h5'
    # 需要保存weight用这一句
    save_model_path = save_model_path
    save_model_details_path = save_model_details_path
    plot_path = save_plot_path
    resize_width = resize_width
    resize_height = resize_height

    # 学习率和轮数
    # INIT_LR = 0.01
    # EPOCHS = 100
    # BATCH_SIZE = 32

    print("[INFO] 开始读取数据")
    (trainX, valX, testX, trainY, valY, testY) = load_data(image_path, resize_width, resize_height)

    # 转换标签，one-hot格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    valY = lb.transform(valY)

    print("[INFO] 建立模型")
    model = SimpleNet.build(width=resize_width, height=resize_height, depth=3,
                               classes=len(lb.classes_))

    print("[INFO] 准备训练网络...")
    H = train_model(trainX, trainY, valX, valY, EPOCHS, BATCH_SIZE, model, INIT_LR, save_model_path)
    print("[INFO] 正在评估模型")
    evaluate_model(model, EPOCHS, testX, testY, H, BATCH_SIZE, lb, plot_path)
    # 保存模型到本地，上面已经保存权重了
    # print("[INFO] 正在保存模型")
    # model.save(model_path)

    # 保存标签对应关系至本地
    print("[INFO] 正在标签")
    saved_model_details = SavedModelDetails(label_encoder=lb, resize_width=resize_width, resize_height=resize_height)
    with open(save_model_details_path, 'wb') as f:
        f.write(pickle.dumps(saved_model_details))

 # 学习率和轮数
INIT_LR = 0.01
EPOCHS = 100
BATCH_SIZE = 32


if __name__ == '__main__':
    image_path = r'./data/cat_dog_image'
    save_model_path = r'save_models/models/cat_dog_weights_simple.h5'
    save_model_details_path = r'save_models/details/cat_dog_simple_details.joblib'
    save_plot_path = r'train_plot/cat_dog_plot_simple.png'
    resize_width = 64
    resize_height = 64

    # 正式运行
    train_process(image_path=image_path, save_model_path=save_model_path,
                  save_model_details_path=save_model_details_path, save_plot_path=save_plot_path,
                  resize_width=resize_width, resize_height=resize_height)
