train_img_dir = "../data/train/"
train_label_file = "../data/train_ship_segmentations.csv"
new_train_label_file = "new_train_ship_segmentations.csv"
no_img_drop_rate = 0.9
test_img_dir = "../data/test/"
test_label_file = "../data/test_ship_segmentations.csv"
valid_ratio = 0.2
batch_size = 20
learning_rate = 0.0001
decay_rate = 0.999
saved_model = "./saved_model_2/model_1600_56126"
img_split_parts = 9
weight_ratio = 15 # must larger than 8


model_flag = "predict" # predict, test, data_exploration, train
continue_train_learning_rate = 0.00005


# predict
predict_batch_size = 20
