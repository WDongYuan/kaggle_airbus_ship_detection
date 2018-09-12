import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import tqdm
import torch.nn as nn
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib
from skimage.morphology import label
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from PIL import Image

from data_processing import *
from model import *
from util import *

if __name__=="__main__":
	transform_list = []
	label_transform_list = []

	transform_list.append(rotate_image(90))
	label_transform_list.append(lambda tmp_img: np.rot90(tmp_img,1).copy())

	transform_list.append(rotate_image(180))
	label_transform_list.append(lambda tmp_img: np.rot90(tmp_img,2).copy())

	transform_list.append(rotate_image(270))
	label_transform_list.append(lambda tmp_img: np.rot90(tmp_img,3).copy())

	transform_list.append(change_brightness(0.3))
	label_transform_list.append(lambda tmp_img: tmp_img)

	transform_list.append(change_brightness(0.6))
	label_transform_list.append(lambda tmp_img: tmp_img)

	transform_list.append(change_brightness(1))
	label_transform_list.append(lambda tmp_img: tmp_img)

	to_pil = transforms.ToPILImage(mode=None)

	dataset = ImageData(train_img_dir, train_label_file, transform_list=transform_list, label_transform_list=label_transform_list)

	valid_ids = np.random.choice(len(dataset),int(len(dataset)*valid_ratio))
	train_ids = np.setdiff1d(np.arange(len(dataset)),valid_ids)
	print(train_ids.shape)
	print(valid_ids.shape)

	train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(train_ids))
	valid_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(valid_ids))

	model = UNET()

	counter = 0
	for i_batch, sample_batch in enumerate(train_dataloader):
	#     fig, axarr = plt.subplots(batch_size, 3)
	    fig, axarr = plt.subplots()
	#     print(axarr.shape)
	    for i in range(batch_size):
	        rnd_num = np.random.randint(0,len(transform_list))
	#         axarr[i,0].imshow(to_pil(sample_batch["img"][i]))
	#         axarr[i,1].imshow(sample_batch["label_img"][i])
	#         axarr[i,2].imshow(sample_batch["weight_img"][i])

	        axarr.imshow(sample_batch["weight_img"][i])
	        
	#         matplotlib.image.imsave("../input/"+str(i)+"_weight_img.png",sample_batch["weight_img"][i])
	#         print(len(np.where(sample_batch["label_img"][i]>0)[0]))
	#         print(len(np.where(sample_batch["label_img"][i]==1)[0]))
	#     prob = model(sample_batch["img"])
	#     print(prob.size())
	    counter += 1
	    if counter>=5:
	        break