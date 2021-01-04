# 导入所需工具包
from keras.models import load_model
import pickle
import cv2
from keras import backend as K
from matplotlib import pyplot as plt
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
from utils.model_utils.build_net import SimpleVGGNet


def image_format(image, resize_width, resize_height):
    image = cv2.resize(image, (resize_width, resize_height))
    image = image.astype("float") / 255.0
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
    model = SimpleVGGNet.build(width=resize_width, height=resize_height, depth=3,
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


# 初始化的时候就加载好权重和模型信息
predict_model, predict_model_details = create_prediction(r'save_models/models/cat_dog_weights_vgg.h5',
                                                         r'save_models/details/cat_dog_vgg_details.joblib')

if __name__ == '__main__':
    # 第一个参数为图片地址，第二参数为模型权重，第三个参数为模型信息,这是预测
    predictions = []
    predictions.append(get_prediction(r'./data/test_image/dogs_00011.jpg', predict_model, predict_model_details))
    predictions.append(get_prediction(r'./data/test_image/panda_00010.jpg', predict_model, predict_model_details))
    predictions.append(get_prediction(r'./data/test_image/panda02.jpg', predict_model, predict_model_details))
    print(predictions)

    # 这是看第一层卷积后的图
    layer_output_show(r'./data/test_image/panda_00022_0.jpg', 10, predict_model, predict_model_details)
    pass
