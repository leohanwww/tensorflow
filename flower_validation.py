# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:50:57 2019
处理flower_pthtos数据集
@author: leohanwww
"""
import os
import os.path
import glob
import numpy as np

INPUT_DATA = 'd:/tensorflow/flower_photos'
OUTPUT_DATA = 'd:/tensorflow/flower_processed_data.npy'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

#读取数据并将数据分割成训练数据,验证数据和测试数据
def create_image_lists(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]#获取各子目录的名字
    is_root_dir = True
    #初始化各数据集
    training_images = {}
    training_labels = {}
    validation_images = {}
    validation_labels = {}
    testing_images = {}
    testing_labels = {}
    current_label = 0
    
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir= False
            continue
        
    #获取一个子目录中所有图片文件
    extensions = ['jpg','jpeg','JPG','JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    for extension in extensions:
        file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
        file_list.extend(glob.glob(file_glob))
        if not file_list: continue
    
        for file_name in file_list:
            image_raw_data = gfile.FastGFile
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    