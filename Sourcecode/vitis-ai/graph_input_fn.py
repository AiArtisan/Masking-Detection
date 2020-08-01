import cv2
import numpy as np
import os
import gzip
import struct

sorts_list = ['mask','nomask']
img_size_net = 160
CONV_INPUT = "x_input"
calib_batch_size = 2500

def load_valid_data(data_path):
    label_cnt = 0
    test_images = []
    test_lables = []
    for sort_path in sorts_list:    
        flower_list = os.listdir(data_path + sort_path)
        for img_name in flower_list:
            img_path = data_path + sort_path + "/" + img_name
            img = cv2.imread(img_path)  
            img_scale = cv2.resize(img,(img_size_net, img_size_net), interpolation = cv2.INTER_CUBIC)
            if not img is None:
                test_images.append(img_scale / 255.)
                test_lables.append(label_cnt)
        label_cnt += 1             
    return test_images, test_lables

dataset_valid_path = './dataset_valid/'
(validSet_images, validSet_lables) = load_valid_data(dataset_valid_path)
validSet_images = np.array(validSet_images)
validSet_lables = np.array(validSet_lables)
def calib_input(iter):
    images = []
    for index in range(0, calib_batch_size):
        images.append(validSet_images[index])

    return {CONV_INPUT: images}

