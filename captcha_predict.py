import torch
import numpy as np
from torch.autograd import Variable
import captcha_generate
import dataset_process
from network import Net
import one_hot


def main():
    net = Net()
    net.eval()
    net.load_state_dict(torch.load('model.pkl'))
    print("load net.")

    test_dataloader = dataset_process.get_predict_data_loader()

    for i, (images, labels) in enumerate(test_dataloader):
        vimage = Variable(images)
        predict_label = net(vimage)
        c0 = captcha_generate.ALL_CHAR_SET[
            np.argmax(predict_label[0, 0:captcha_generate.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_generate.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_generate.ALL_CHAR_SET_LEN:2 * captcha_generate.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_generate.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_generate.ALL_CHAR_SET_LEN:3 * captcha_generate.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_generate.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_generate.ALL_CHAR_SET_LEN:4 * captcha_generate.ALL_CHAR_SET_LEN].data.numpy())]

        true_label = one_hot.decode(labels.numpy()[0])
        print('predict--> %s%s%s%s' % (c0, c1, c2, c3), 'True-->%s' % true_label)


if __name__ == '__main__':
    main()
