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
    if torch.cuda.is_available():
        net = net.cuda()
    test_dataloader = dataset_process.get_test_data_loader()

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_dataloader, 0):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels
        # print(images)
        predict_labels1, predict_labels2, predict_labels3, predict_labels4 = net(images)
        c1 = captcha_generate.ALL_CHAR_SET[predict_labels1.topk(1, dim=1)[1]]
        c2 = captcha_generate.ALL_CHAR_SET[predict_labels2.topk(1, dim=1)[1]]
        c3 = captcha_generate.ALL_CHAR_SET[predict_labels3.topk(1, dim=1)[1]]
        c4 = captcha_generate.ALL_CHAR_SET[predict_labels4.topk(1, dim=1)[1]]
        predict_label = '%s%s%s%s' % (c1, c2, c3, c4)
        
        true_label = one_hot.decode(labels.numpy()[0])
        print(predict_label,'-------->',true_label)
        total += labels.size(0)
        if (predict_label == true_label):
            correct += 1
        if (total % 200 == 0):
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

if __name__ == '__main__':
    main()
