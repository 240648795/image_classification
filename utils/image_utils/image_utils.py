import os
import sys
import cv2 as cv
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def save_img(video_path, pic_save_path, width, height, pixel_num):  # 提取视频中图片 按照每帧提取
    video_path_ = video_path  # 视频所在的路径
    pic_save_path_ = pic_save_path  # 保存图片的上级目录
    videos = os.listdir(video_path_)  # 返回指定路径下的文件和文件夹列表。
    for video_name in videos:  # 依次读取视频文件
        file_name = video_name.split('.')[0]  # 拆分视频文件名称 ，剔除后缀
        folder_name = pic_save_path_ + file_name  # 保存图片的上级目录+对应每条视频名称 构成新的目录存放每个视频的
        os.makedirs(folder_name, exist_ok=True)  # 创建存放视频的对应目录
        vc = cv.VideoCapture(video_path_ + video_name)  # 读入视频文件
        c = 0  # 计数  统计对应帧号
        rval = vc.isOpened()  # 判断视频是否打开  返回True或Flase

        while rval:  # 循环读取视频帧
            rval, frame = vc.read()  # videoCapture.read() 函数，第一个返回值为是否成功获取视频帧，第二个返回值为返回的视频帧：
            pic_path = folder_name + '/'
            if rval:
                if c % pixel_num == 0:
                    image = frame.copy()
                    image_height = image.shape[0]
                    image_width = image.shape[1]
                    image_left = image[0:image_height, 0:image_width // 2, :]
                    image_left_resized = cv.resize(image_left, (width, height), interpolation=cv.INTER_AREA)
                    image_right = image[0:image_height, image_width // 2:image_width, :]
                    image_right_resized = cv.resize(image_right, (width, height), interpolation=cv.INTER_AREA)
                    cv.imwrite(pic_path + file_name + '_left_' + str(c) + '.jpg', image_left_resized)
                    cv.imwrite(pic_path + file_name + '_right_' + str(c) + '.jpg', image_right_resized)
                    cv.waitKey(1)  # waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下 键,则接续等待(循环)
                c = c + 1
            else:
                break
        vc.release()
        print('save_success' + folder_name)


class Augmentation(object):
    def __init__(self, img_type="png"):
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

    def augmentation(self, picname, savedir, pre, img_num):
        print("运行 Augmentation")
        # Start augmentation.....
        img_t = load_img(picname)  # 读入train

        x_t = img_to_array(img_t)  # 转换成矩阵
        img = x_t
        img = img.reshape((1,) + img.shape)
        print(img.shape)

        if not os.path.lexists(savedir):
            os.mkdir(savedir)
        print("running %d doAugmenttaion" % 0)
        self.do_augmentate(img, savedir, pre, imgnum=img_num)  # 数据增强

    def do_augmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='jpg', imgnum=10):
        # augmentate one image
        datagen = self.datagen
        i = 0
        for _ in datagen.flow(
                img,
                batch_size=batch_size,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format):
            i += 1
            if i >= imgnum:
                break

    def batchgenerate(self, picsrc, savedir, imgnum):
        picList = os.listdir(picsrc)
        len1 = len(picList)
        subfile = "SUB"
        if not os.path.lexists("SUB"):
            os.mkdir("SUB")
        for picname in picList:
            file_src = picsrc + '/' + picname
            pre = picname.split('.')[0]
            self.augmentation(file_src, "SUB", pre, imgnum)
        picListsub = os.listdir("SUB")
        num = []
        for i in range(len1):
            num.append(0)
        print("图片个数", len(num))
        loc = 0
        print("转换图片列表", picList)
        for picname in picListsub:
            file_src = "SUB" + '/' + picname
            pre = picname.split('_')[0] + '_' + picname.split('_')[1] + '.jpg'
            location = picList.index(pre)
            all = picname.split('_')[0] + '_' + picname.split('_')[1] + "_" + str(num[location]) + '.jpg'
            num[location] = num[location] + 1

            file_dst = savedir + '/' + all
            shutil.copyfile(file_src, file_dst)
            os.remove(file_src)

        # 删除文件夹，可以注释起来看结果对不对
        os.rmdir(subfile)


if __name__ == '__main__':
    # 将视频转化为图片
    # save_img(r'../../data/video/', r'../data/exp_image/', 64, 32, 5)

    # 按照类别生成随即图片
    base_dir = os.path.join(sys.path[1], '')
    pic_path = base_dir + 'data/cat_dog_image/'
    save_path = base_dir + 'data/cat_dog_generate/'
    for root, dirs, files in os.walk(pic_path):
        for dir in dirs:
            picsrc = root + dir
            savedir = save_path + dir
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            aug = Augmentation()
            aug.batchgenerate(picsrc, savedir, 5)

    # 单次变换图片的程序
    # base_dir = os.path.join(sys.path[1], '')
    # picsrc = base_dir + 'data/cat_dog_image/test'
    # savedir = base_dir + 'data/cat_dog_generate'
    # aug = Augmentation()
    # aug.batchgenerate(picsrc, savedir, 2)
