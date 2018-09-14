import os
import numpy as np
from torchvision import transforms, utils
from PIL import Image
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib

# rle encode and decode

def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


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
        
        w_img = np.multiply(10-w_img,img)+1
        #w_img[np.nonzero(w_img)] = 1
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

def classify_accuracy(prob,true_label,dim=1):
    #print(prob.shape)
    pred_label = np.argmax(prob,axis=dim)
    print(pred_label.sum())
    #print(pred_label.shape)
    tp = np.multiply(pred_label,true_label)
    print(tp.sum())
    print(true_label.sum())
    print(tp.sum()/true_label.sum())
    return tp.sum()/true_label.sum()

def save_arr_as_img(img_as_arr,path):
     matplotlib.image.imsave(path,img_as_arr)


