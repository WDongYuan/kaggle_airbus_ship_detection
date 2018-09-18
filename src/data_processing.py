import os
import numpy as np
from torchvision import transforms, utils
import torch
from PIL import Image
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib
from util import *

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
        
        w_img = np.multiply(weight_ratio-w_img,img)+1

        #w_img[np.nonzero(w_img)] = 1
        return w_img

    def get_img_part(self,img,r,c,parts,i):
        num = int(np.sqrt(parts))
        return img[int(i/num)*int((r/num)):(int(i/num)+1)*int((r/num)),
            (i%num)*int((c/num)):(i%num+1)*int((c/num))]

    def crop_img(self,img,label_img,parts):
        num = np.sqrt(parts)
        r,c = label_img.shape
        max_idx = 0
        max_val = 0
        for i in range(parts):
            tmp_img = self.get_img_part(label_img,r,c,parts,i)
            if max_val<tmp_img.sum():
                max_val = tmp_img.sum()
                max_idx = i
        return self.get_img_part(img,r,c,parts,max_idx),self.get_img_part(label_img,r,c,parts,max_idx)
    
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        tmp_img = Image.open(self.img_dir+self.img_list[idx])
        rle_0 = self.label.query('ImageId=="'+self.img_list[idx]+'"')['EncodedPixels']
        label_img = masks_as_image(rle_0)[:,:,0]

        tmp_img,label_img = self.crop_img(np.array(tmp_img),label_img,img_split_parts)
        tmp_img = Image.fromarray(tmp_img)

        if self.transform_list is not None:
            rnd_num = np.random.randint(0,len(self.transform_list))
        if self.transform_list is not None:
            tmp_img = self.transform_list[rnd_num](tmp_img)
        tmp_img = transforms.functional.to_tensor(tmp_img)
        
        
        if self.label_transform_list is not None:
            label_img = self.label_transform_list[rnd_num](label_img)
        w_label_img = self.WeightLabelImg(label_img)
        
        return {"img_name": self.img_list[idx], "img": tmp_img, "label_img": label_img, "weight_img": w_label_img}

class TestImageData:
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)

    def get_img_part(self,img,r,c,parts,i):
        num = int(np.sqrt(parts))
        return img[int(i/num)*int((r/num)):(int(i/num)+1)*int((r/num)),
            (i%num)*int((c/num)):(i%num+1)*int((c/num))]

    def crop_img(self,img,parts):
        num = np.sqrt(parts)
        print(img.shape)
        assert 1==2
        r,c,_ = img.shape
        sub_imgs = []
        for i in range(parts):
            tmp_img = self.get_img_part(img,r,c,parts,i)
            sub_imgs.append(tmp_img)
        sub_imgs = np.array(sub_imgs)
        return sub_imgs
    
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        tmp_img = Image.open(self.img_dir+self.img_list[idx])
        ori_img = transforms.functional.to_tensor(tmp_img)
        if img_split_parts==1:
            tmp_img = transforms.functional.to_tensor(tmp_img)
        else:
            tmp_img = self.crop_img(np.array(tmp_img),img_split_parts)
            # tmp_img = Image.fromarray(tmp_img)
            # tmp_img = transforms.functional.to_tensor(tmp_img)
            tmp_img = torch.from_numpy(tmp_img).permute(0,3,1,2).float()

            # tmp_img = self.get_img_part(np.array(tmp_img),768,768,img_split_parts,5)
            # tmp_img = Image.fromarray(tmp_img)
            # tmp_img = transforms.functional.to_tensor(tmp_img)
            # tmp_img = tmp_img.unsqueeze(0).expand(9,-1,-1,-1)

        return {"img_name": self.img_list[idx], "img": tmp_img, "ori_img": ori_img}

def change_brightness(factor):
    def func(img):
        return transforms.functional.adjust_brightness(img, factor)
    return func

def rotate_image(angle):
    def func(img):
        return transforms.functional.rotate(img,angle)
    return func

def classify_accuracy(prob,true_label,dim=1):
    pred_label = np.argmax(prob,axis=dim)
    tp = np.multiply(pred_label,true_label)
    print("precision: %f"%(tp.sum()/(pred_label.sum()+1)), end=" | ")
    print("recall: %f"%(tp.sum()/(true_label.sum()+1)))
    return tp.sum()/(true_label.sum()+1)

def save_arr_as_img(img_as_arr,path):
     matplotlib.image.imsave(path,img_as_arr)

def combine_image_parts(imgs):
    parts,h,w = imgs.shape
    base = int(np.sqrt(parts))
    new_img = np.vstack([np.hstack([imgs[r*base+c] for c in range(base)]) for r in range(base)])
    return new_img


