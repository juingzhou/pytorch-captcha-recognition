import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import one_hot
import captcha_generate


class mydataset(Dataset):

    def __init__(self, folder, dataset_transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]

        self.transforms = dataset_transform

    def __len__(self):
        return (len(self.train_image_file_paths))

    def __getitem__(self, index):
        image_root = self.train_image_file_paths[index]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transforms is not None:
            image = self.transforms(image)

        label = one_hot.encode(image_name.split('_')[
                                   0])  # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理

        return image, label


dataset_transform = transforms.Compose([
    # transforms.ColorJitter(),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_train_data_loader():
    dataset = mydataset(captcha_generate.TRAIN_DATASET_PATH, dataset_transform=dataset_transform)
    return DataLoader(dataset, batch_size=captcha_generate.BATCH_SIZE, shuffle=True, num_workers=2)


def get_test_data_loader():
    dataset = mydataset(captcha_generate.TEST_DATASET_PATH, dataset_transform=dataset_transform)
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)


def get_predict_data_loader():
    dataset = mydataset(captcha_generate.PREDICT_DATASET_PATH, dataset_transform=dataset_transform)
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
