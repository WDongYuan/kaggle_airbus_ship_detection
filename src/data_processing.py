import os
import numpy as np
from torchvision import transforms, utils
from PIL import Image
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

class ImageData:
    def __init__(self, img_dir, label_file, transform_list=None, label_transform_list=None):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.label_file = label_file
        self.label = pd.read_csv(label_file)
        self.transform_list = transform_list
        self.label_transform_list = label_transform_list
    def WeightLabelImg(self,img):
        h,w = img.shape
        w_img = np.zeros((h,w))
        w_img[0:h-1,:] += img[1:h] #down
        w_img[1:h,:] += img[0:h-1] #up
        w_img[:,0:w-1] += img[:,1:w] #right
        w_img[:,1:w] += img[:,0:w-1] #left
        
        w_img[0:h-1,0:w-1] += img[1:h,1:w] #down right
        w_img[1:h,1:w] += img[0:h-1,0:w-1] #up left
        w_img[1:h,0:w-1] += img[0:h-1,1:w] #up right
        w_img[0:h-1:,1:w] += img[1:h,0:w-1] #down left
        
        w_img = np.multiply(8-w_img,img)
        w_img[np.nonzero(w_img)] = 1
        return w_img
    
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
#         tmp_img = imread(self.img_dir+self.img_list[idx])
        tmp_img = Image.open(self.img_dir+self.img_list[idx])
        rnd_num = np.random.randint(0,len(self.transform_list))
#         rnd_num = 5
        if self.transform_list is not None:
            tmp_img = self.transform_list[rnd_num](tmp_img)
        tmp_img = transforms.functional.to_tensor(tmp_img)
        
        rle_0 = self.label.query('ImageId=="'+self.img_list[idx]+'"')['EncodedPixels']
        label_img = masks_as_image(rle_0)[:,:,0]
        if self.label_transform_list is not None:
            label_img = self.label_transform_list[rnd_num](label_img)
        w_label_img = self.WeightLabelImg(label_img)
#         label_img = transforms.functional.to_tensor(label_img)
        
        return {"img_name": self.img_list[idx], "img": tmp_img, "label_img": label_img, "weight_img": w_label_img}
def change_brightness(factor):
    def func(img):
        return transforms.functional.adjust_brightness(img, factor)
    return func

def rotate_image(angle):
    def func(img):
        return transforms.functional.rotate(img,angle)
    return func