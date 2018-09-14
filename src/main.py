import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib
from skimage.morphology import label
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from PIL import Image
import time

from data_processing import *
from model import *
from util import *
def ModelPredict(model,valid_dataloader):
	print("Making prediction...")
	model.eval()
	for i_batch, sample_batch in enumerate(valid_dataloader):
		log_prob = model(sample_batch["img"].cuda())
		predict_label = np.argmax(log_prob.data.cpu().numpy(),axis=1)
		save_arr_as_img(predict_label[0],"./test_dir/predict_"+str(i_batch)+".png")
		return

def TrainModel(model, optimizer, train_dataloader, valid_dataloader, decay_step,decay_rate, total_epoch, lr):

	model.cuda()

	loss_func = nn.NLLLoss()
	for epoch in range(total_epoch):
		print("##################################")
		print("epoch "+str(epoch))
		loss_mean = 0
		batch_count = 0
		# start_time = time.time()
		for i_batch, sample_batch in enumerate(train_dataloader):
			# print(time.time()-start_time)
			# start_time = time.time()

			optimizer.zero_grad()
			log_prob = model(sample_batch["img"].cuda())
			#print(log_prob.size())
			#print(sample_batch["weight_img"].size())
			log_prob = torch.mul(log_prob,sample_batch["weight_img"].unsqueeze(1).float().cuda())
			loss = loss_func(log_prob,sample_batch["label_img"].long().cuda())
			# loss_mean = loss.mean()
			print("%f,%f,%d"%(loss.mean().data.cpu().numpy(),classify_accuracy(log_prob.data.cpu().numpy(),
				sample_batch["label_img"].numpy()),(batch_count+1)*batch_size),end="\n", flush=True)
			
			# save_arr_as_img(np.argmax(log_prob.data.cpu().numpy()[0],axis=0),"./test_dir/prob_"+str(batch_count)+".png")
			# save_arr_as_img(sample_batch["weight_img"].numpy()[0],"./test_dir/label_"+str(batch_count)+".png")
			loss.backward()
			optimizer.step()

			# print(time.time()-start_time)
			lr *= 0.995
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
			print("############################")

			if batch_count%100==0:
				torch.save(model,"./saved_model/model_"+str(batch_count))
			batch_count += 1
			# if batch_count>=20:
			# 	return
			# start_time = time.time()
		# print(loss_mean/batch_count)

	
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





	# dataset = ImageData(train_img_dir, train_label_file, transform_list=transform_list, label_transform_list=label_transform_list)
	dataset = ImageData(train_img_dir, train_label_file, transform_list=None, label_transform_list=None)

	valid_ids = np.random.choice(len(dataset),int(len(dataset)*valid_ratio))
	train_ids = np.setdiff1d(np.arange(len(dataset)),valid_ids)
	# print(train_ids.shape)
	# print(valid_ids.shape)

	train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(train_ids))
	valid_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(valid_ids))
	if train_flag:
		model = UNET()
		optimizer = optim.Adam(model.parameters(),lr = learning_rate)
		TrainModel(model,
			optimizer,
			train_dataloader,
			valid_dataloader,
			decay_step = 0,
			decay_rate = 0,
			total_epoch = 10,
			lr = learning_rate)
	else:
		model = torch.load(saved_model)
		ModelPredict(model,valid_dataloader)

	# counter = 0
	# for i_batch, sample_batch in enumerate(train_dataloader):
	# 	fig, axarr = plt.subplots(batch_size, 3)
	# #     print(axarr.shape)
	# 	for i in range(batch_size):
	# 		rnd_num = np.random.randint(0,len(transform_list))
	# #         axarr[i,0].imshow(to_pil(sample_batch["img"][i]))
	# #         axarr[i,1].imshow(sample_batch["label_img"][i])
	# #         axarr[i,2].imshow(sample_batch["weight_img"][i])

	# #         print(len(np.where(sample_batch["label_img"][i]>0)[0]))
	# #         print(len(np.where(sample_batch["label_img"][i]==1)[0]))
	# #     prob = model(sample_batch["img"])
	# #     print(prob.size())
	# 	matplotlib.image.imsave("./test_dir/"+str(counter)+"_weight_img.png",sample_batch["weight_img"][0])
	# 	counter += 1
	# 	if counter>=20:
	# 		break
