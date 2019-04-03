# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha
from PIL import Image
import random
import time
import os
import string
# 验证码的字符
ALL_CHAR_SET = string.digits + string.ascii_uppercase
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4
BATCH_SIZE = 64
# 图像大小
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 170

TRAIN_DATASET_PATH = 'dataset' + os.path.sep + 'train'
TEST_DATASET_PATH = 'dataset' + os.path.sep + 'test'
PREDICT_DATASET_PATH = 'dataset' + os.path.sep + 'predict'

def random_captcha():
    captcha_text = []
    for i in range(MAX_CAPTCHA):
        c = random.choice(ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)

# 生成字符对应的二维码
def gen_captcha_text_and_image():
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    # 返回对应图片的字符
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

if __name__ == '__main__':
    count = 1000
    path = TEST_DATASET_PATH
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(count):
        now = str(int(time.time()))
        text, image = gen_captcha_text_and_image()
        filename = text + '_' + now + '.png'
        image.save(path + os.path.sep + filename)
        print('save %d: %s'%(i+1, filename))

