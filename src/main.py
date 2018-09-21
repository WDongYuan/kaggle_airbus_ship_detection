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
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from data_processing import *
from model import *
from util import *
def ModelPredict(model,valid_dataloader):
    print("Making prediction...")
    model.eval()
    with torch.no_grad(): 
        for i_batch, sample_batch in enumerate(valid_dataloader):
            sample_batch["img"] = sample_batch["img"].squeeze()
            log_prob = model(sample_batch["img"].cuda())
            classify_accuracy(log_prob.data.cpu().numpy(),sample_batch["label_img"].numpy())
#             predict_label = np.argmax(log_prob.data.cpu().numpy(),axis=1)
#             print(sorted([predict_label[i].sum()/256/256 for i in range(9)]))
            # return
#             for i_img in range(predict_label.shape[0]):
                # save_arr_as_img(predict_label[i_img],"./test_dir/predict_"+str(i_batch)+"_"+str(i_img)+".png")
                # save_arr_as_img(sample_batch["label_img"][i_img].numpy(),"./test_dir/predict_"+str(i_batch)+"_"+str(i_img)+"_true.png")

#                 save_arr_as_img(np.transpose(sample_batch["img"][i_img].int().numpy(),(1,2,0)),"./test_dir/predict_"+str(i_batch)+"_"+str(i_img)+"_true_img.png")
            
            #print(i_batch, end=" ", flush=True)
            if i_batch>=20:
                return

def ModelTest(model,test_dataloader):
    print("Making prediction...")
    model.eval()
    pred_file = []
    start_time = time.time()
    with torch.no_grad(): 
        for i_batch, sample_batch in enumerate(test_dataloader):
#             print("batch %d"%(i_batch))

#             sample_batch["img"] = sample_batch["img"].squeeze()
            b,p,c,h,w = sample_batch["img"].size()
            sample_batch["img"] = sample_batch["img"].view(b*p,c,h,w)
            
            log_prob = model(sample_batch["img"].float().cuda())
            log_prob = log_prob.data.cpu().numpy()
            predict_label = np.argmax(log_prob,axis=1)
            predict_label = predict_label.reshape((b,p,h,w))

            for i in range(b):
                big_pred_img = combine_image_parts(predict_label[i])
                img_rle = rle_encode(big_pred_img)
                img_rle = img_rle if len(img_rle)>0 else None
                pred_file += [{'ImageId': sample_batch["img_name"][i], 'EncodedPixels': img_rle}]
                
#                 save_arr_as_img(big_pred_img,"./test_dir/big_predict_"+str(i_batch)+"_"+str(i)+".png")
#                 save_arr_as_img(np.transpose(predict_label[i],(1,2,0)),"./test_dir/predict_"+str(i_batch)+"_"+str(i)+".png")
#                 save_arr_as_img(np.transpose(sample_batch["ori_img"][i].numpy(),(1,2,0)),"./test_dir/predict_img_"+str(i_batch)+"_"+str(i)+"_ori.png")

            if (i_batch+1)*predict_batch_size%500==0:
                print("%d images processed. Time:%fs"%((i_batch+1)*predict_batch_size,time.time()-start_time))
                
    submission_df = pd.DataFrame(pred_file)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv('submission.csv', index=False)
#     submission_df.sample(10)


                
            

def TrainModel(model, optimizer, train_dataloader, valid_dataloader, decay_step,decay_rate, total_epoch, lr):

    valid_iter = iter(valid_dataloader)

    model.cuda()

    loss_func = nn.NLLLoss()
    for epoch in range(total_epoch):
        print("epoch "+str(epoch))

        for i_batch, sample_batch in enumerate(train_dataloader):

            optimizer.zero_grad()
            log_prob = model(sample_batch["img"].cuda())
            #print(log_prob.size())
            #print(sample_batch["weight_img"].size())
            log_prob = torch.mul(log_prob,sample_batch["weight_img"].unsqueeze(1).float().cuda())
            loss = loss_func(log_prob,sample_batch["label_img"].long().cuda())
            classify_accuracy(log_prob.data.cpu().numpy(),sample_batch["label_img"].numpy())
            print("loss: %f | %d images processed"%(loss.mean().data.cpu().numpy(),(i_batch+1)*batch_size),end="\n", flush=True)

            loss.backward()
            optimizer.step()

            lr *= decay_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if i_batch%200==0:
                print("saving model to ./saved_model/model_"+str(i_batch)+"_"+str(int(time.time()%100000)))
                torch.save(model,"./saved_model/model_"+str(i_batch)+"_"+str(int(time.time()%100000)))

#             if i_batch%50==0:
#                 print("validating model...")
#                 model.eval()
#                 for i in range(5):
#                     valid_batch = next(valid_iter)
#                     valid_prob = model(valid_batch["img"].cuda())
#                     classify_accuracy(valid_prob.data.cpu().numpy(),valid_batch["label_img"].numpy())
#                 model.train()

            print("############################")


    
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


    print("Dropping images without ship...")
    drop_no_ship_img(train_label_file,new_train_label_file,no_img_drop_rate)

#     dataset = ImageData(train_img_dir, new_train_label_file, transform_list=transform_list, label_transform_list=label_transform_list)
    dataset = ImageData(train_img_dir, new_train_label_file, transform_list=None, label_transform_list=None)
#     dataset = ImageData(test_img_dir, test_label_file, transform_list=None, label_transform_list=None)

    valid_ids = np.random.choice(len(dataset),int(len(dataset)*valid_ratio))
    train_ids = np.setdiff1d(np.arange(len(dataset)),valid_ids)
    # print(train_ids.shape)
    # print(valid_ids.shape)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(train_ids))
    valid_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(valid_ids))


    test_dataset = TestImageData(test_img_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=predict_batch_size, shuffle=True, num_workers=0)

    if model_flag == "train":
        model = UNET()
        optimizer = optim.Adam(model.parameters(),lr = learning_rate)
        TrainModel(model,
            optimizer,
            train_dataloader,
            valid_dataloader,
            decay_step = 0,
            decay_rate = decay_rate,
            total_epoch = 10,
            lr = learning_rate)
    elif model_flag == "continue_train":
        model = torch.load(saved_model)
        optimizer = optim.Adam(model.parameters(),lr = continue_train_learning_rate)
        TrainModel(model,
            optimizer,
            train_dataloader,
            valid_dataloader,
            decay_step = 0,
            decay_rate = decay_rate,
            total_epoch = 10,
            lr = continue_train_learning_rate)

    ## This is a prediction process for only the selected part (the part with the most labeled pixels) of a image
    elif model_flag == "predict":
        model = torch.load(saved_model)
        ModelPredict(model,valid_dataloader)

    elif model_flag == "test":
        model = torch.load(saved_model)
        ModelTest(model,test_dataloader)
    
    elif model_flag == "data_exploration":
#         data_exploration(train_img_dir, train_label_file)
        drop_no_ship_img(train_label_file,new_train_label_file,no_img_drop_rate)

    # counter = 0
    # for i_batch, sample_batch in enumerate(train_dataloader):
    #     fig, axarr = plt.subplots(batch_size, 3)
    # #     print(axarr.shape)
    #     for i in range(batch_size):
    #         rnd_num = np.random.randint(0,len(transform_list))
    # #         axarr[i,0].imshow(to_pil(sample_batch["img"][i]))
    # #         axarr[i,1].imshow(sample_batch["label_img"][i])
    # #         axarr[i,2].imshow(sample_batch["weight_img"][i])

    # #         print(len(np.where(sample_batch["label_img"][i]>0)[0]))
    # #         print(len(np.where(sample_batch["label_img"][i]==1)[0]))
    # #     prob = model(sample_batch["img"])
    # #     print(prob.size())
    #     matplotlib.image.imsave("./test_dir/"+str(counter)+"_weight_img.png",sample_batch["weight_img"][0])
    #     counter += 1
    #     if counter>=20:
    #         break
