import os
import torch
from .load_train import Spoofing_train
from .load_valtest import Spoofing_valtest


class dataset_info(object):

    def __init__(self):
        self.root_dir = ""


class data_merge(object):

    def __init__(self, image_dir):
        self.dic = {}
        self.image_dir = image_dir

        for i in os.listdir(self.image_dir):
            info = dataset_info()
            info.root_dir = self.image_dir+'/'+i
            self.dic[i] = info

    def get_single_dataset(self, data_name="", train=True, img_size=256, transform=None, UUID=-1):
        if train:
            data_dir = self.dic[data_name].root_dir
            data_set = Spoofing_train(os.path.join(data_dir, "label.txt"), os.path.join(data_dir, "image"), transform=transform, img_size=img_size, UUID=UUID)
        else:
            data_dir = self.dic[data_name].root_dir
            data_set = Spoofing_valtest(os.path.join(data_dir, "label.txt"), os.path.join(data_dir, "image"), transform=transform, img_size=img_size, UUID=UUID)

        print("Loading {}, number: {}".format(data_name, len(data_set)))
        return data_set

    def get_datasets(self, train=True, img_size=256, transform=None):
        data_name_list_train = [i for i in list(self.dic.keys())]

        sum_n = 0
        if train:
            data_set_sum = self.get_single_dataset(data_name=data_name_list_train[0], train=True, img_size=img_size, transform=transform, UUID=0)
            sum_n = len(data_set_sum)
            for i in range(1, len(data_name_list_train)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_train[i], train=True, img_size=img_size, transform=transform, UUID=i)
                data_set_sum += data_tmp
                sum_n += len(data_tmp)

        print("Total number: {}".format(sum_n))
        return data_set_sum
