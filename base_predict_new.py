# 导入所需工具包
# 已弃用，用来学习
from keras.models import load_model
import argparse
import pickle
import cv2

def image_format(image,resize_width,resize_height):
	image = cv2.resize(image, (resize_width, resize_height)).flatten()

	# scale the pixel values to [0, 1]
	image = image.astype("float") / 255.0

	# 是否要对图像就行拉平操作
	# image = image.flatten()
	image = image.reshape((1, image.shape[0]))
	return image


# CNN的时候需要原始图像
# image = image.reshape((1, image.shape[0], image.shape[1],
# 		image.shape[2]))

if __name__ == '__main__':
	image_path= r'./data/exp_image/shashi/shashi_left_0.jpg'
	model_path=r'./save_models//models/base_train_new_model.h5'
	model_details_path=r'./save_models/details/base_train_new_details.joblib'
	resize_width = 32
	resize_height = 32

	# 加载测试数据并进行相同预处理操作
	image = cv2.imread(image_path)
	image = image_format(image,resize_width,resize_height)

	# 读取模型和标签
	print("[INFO] loading network and label binarizer...")
	model = load_model(model_path)
	lb = pickle.loads(open(model_details_path, "rb").read())

	# 预测
	preds = model.predict(image)

	# 得到预测结果以及其对应的标签
	i = preds.argmax(axis=1)[0]
	label = lb.classes_[i]

	print (label)

