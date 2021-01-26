# 导入所需工具包
from keras.models import load_model
import pickle
import cv2
from keras import backend as K
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
from utils.model_utils.build_net import SimpleVGGNet, SimpleNet


def image_format(image, resize_width, resize_height):
    image_org = cv2.resize(image, (resize_width, resize_height))


    img_org_hsv = cv2.cvtColor(image_org, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_org_hsv)
    s_list = [0 for i in range(0, s.shape[0] * s.shape[1])]
    s_list_np = np.asarray(s_list).reshape(s.shape).astype(s.dtype)
    v_list = [0 for i in range(0, v.shape[0] * v.shape[1])]
    v_list_np = np.asarray(v_list).reshape(v.shape).astype(v.dtype)
    img_org_hsv_trans = cv2.merge([h, s_list_np, v_list_np])


    image = img_org_hsv_trans.astype("float") / 255.0
    image = image.reshape((1, image.shape[0], image.shape[1],
                           image.shape[2]))
    return image


def create_prediction(model_path, model_details_path):
    # 读取标签和其他细节
    print("[INFO] loading label binarizer and details...")
    saved_model_details = pickle.loads(open(model_details_path, "rb").read())
    lb = saved_model_details.get_label_encoder()
    resize_width = saved_model_details.get_resize_width()
    resize_height = saved_model_details.get_resize_height()
    # 这里可以切换SimpleVGGNet和SimpleNet
    model = SimpleNet.build(width=resize_width, height=resize_height, depth=3,
                               classes=len(lb.classes_))
    model.load_weights(model_path)
    return model, saved_model_details


def get_prediction(image_path, model, saved_model_details):
    lb = saved_model_details.get_label_encoder()
    resize_width = saved_model_details.get_resize_width()
    resize_height = saved_model_details.get_resize_height()
    # 加载测试数据并进行相同预处理操作
    image = cv2.imread(image_path)
    image = image_format(image, resize_width, resize_height)
    # 预测
    preds = model.predict(image)
    # 得到预测结果以及其对应的标签
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    return label


def get_prediction_image(image_org, model, saved_model_details, threshold):
    lb = saved_model_details.get_label_encoder()
    resize_width = saved_model_details.get_resize_width()
    resize_height = saved_model_details.get_resize_height()

    image = image_format(image_org, resize_width, resize_height)
    # 预测
    preds = model.predict(image)
    # 得到预测结果以及其对应的标签

    # i = preds.argmax(axis=1)[0]
    # label = lb.classes_[i]
    label = 'unknown'
    if preds.argmax(axis=1) >= threshold:
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]
    return label


def layer_output_show(image_path, layer_num, model, saved_model_details):
    resize_width = saved_model_details.get_resize_width()
    resize_height = saved_model_details.get_resize_height()
    # 加载测试数据并进行相同预处理操作
    image = cv2.imread(image_path)
    image = image_format(image, resize_width, resize_height)

    # 第一个 model.layers[0],不修改,表示输入数据；
    # 第二个 model.layers[ ],修改为需要输出的层数的编号[]
    layer_1 = K.function([model.layers[0].input], [model.layers[layer_num].output])

    # 只修改input_image
    f1 = layer_1([image])[0]
    sample_num = f1.shape[0]
    sample_width = f1.shape[1]
    sample_height = f1.shape[2]
    features_num = f1.shape[3]
    # 第一层卷积后的特征图展示，输出是（sample_num,sample_width,sample_height,features_num），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
    for _ in range(features_num):
        show_img = f1[:, :, :, _]
        show_img.shape = [sample_width, sample_height]
        # 一共四行，每行得列数由特征图得总个数决定
        plt.subplot(4, features_num // 4, _ + 1)
        plt.imshow(show_img)
        plt.axis('off')
    plt.show()


def two_step_prediction(file_path, need_width, need_height, threshold):
    img_org = cv2.imread(file_path)
    img_org = cv2.resize(img_org, (400, 400))
    cols = img_org.shape[1]
    rows = img_org.shape[0]
    need_cols = cols // need_width
    need_rows = rows // need_height

    predicts = []

    for col_i in range(0, need_cols):
        for row_i in range(0, need_rows):
            each_col_start = need_width * col_i
            each_col_end = need_width * col_i + need_width
            each_row_start = need_height * row_i
            each_row_end = need_height * row_i + need_height
            each_pic = img_org[each_row_start:each_row_end, each_col_start:each_col_end, :]
            label = get_prediction_image(each_pic, predict_model, predict_model_details, threshold)
            predicts.append(label)
            if label != 'unknown':
                if label == 'yellow_earth':
                    cv2.rectangle(img_org, (each_col_start, each_row_start), (each_col_end, each_row_end),
                                  (255, 255, 255), 3)
                    cv2.putText(img_org, label, (each_col_start + 5, each_row_start + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 255), 2)
                elif label == 'red_earth':
                    cv2.rectangle(img_org, (each_col_start, each_row_start), (each_col_end, each_row_end),
                                  (255, 255, 255), 3)
                    cv2.putText(img_org, label, (each_col_start + 5, each_row_start + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 0, 255), 2)
                elif label == 'coal':
                    cv2.rectangle(img_org, (each_col_start, each_row_start), (each_col_end, each_row_end),
                                  (255, 255, 255), 3)
                    cv2.putText(img_org, label, (each_col_start + 5, each_row_start + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 0), 2)
                elif label == 'stone':
                    cv2.rectangle(img_org, (each_col_start, each_row_start), (each_col_end, each_row_end),
                                  (255, 255, 255), 3)
                    cv2.putText(img_org, label, (each_col_start + 5, each_row_start + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 0, 0), 2)

    print(predicts)
    cv2.imshow('work_condition_recognition', img_org)
    cv2.waitKey(0)


# 初始化的时候就加载好权重和模型信息
predict_model, predict_model_details = create_prediction(r'save_models/models/cut_material_image_vgg.h5',
                                                         r'save_models/details/cut_material_image_vgg_details.joblib')

if __name__ == '__main__':
    # 第一个参数为图片地址，第二参数为模型权重，第三个参数为模型信息,这是预测
    # predictions = []
    # predictions.append(get_prediction(r'./data/cut_material_image/coal/cut_image_col_02ab0c0f0-4e58-11eb-b8f4-94e6f7f8d382_row_4.jpg', predict_model, predict_model_details))
    # predictions.append(get_prediction(r'./data/cut_material_image/red_earth/cut_image_col_02aa05034-4e58-11eb-b697-94e6f7f8d382_row_6.jpg', predict_model, predict_model_details))
    # predictions.append(get_prediction(r'./data/cut_material_image/yellow_earth/cut_image_col_047fb6c70-4e5e-11eb-a96f-94e6f7f8d382_row_9.jpg', predict_model, predict_model_details))
    # print(predictions)

    two_step_prediction(r'./data/test_image/stone.jpg', 40, 30, 0.1)

    # predictions = []
    # predictions.append(get_prediction(r'./data/test_image/cut_image_col_1b977042c-4e69-11eb-91ef-94e6f7f8d382_row_4.jpg', predict_model, predict_model_details))
    # predictions.append(get_prediction(r'./data/test_image/cut_image_col_8b98884e8-4e69-11eb-a531-94e6f7f8d382_row_6.jpg', predict_model, predict_model_details))
    # print(predictions)

    pass
