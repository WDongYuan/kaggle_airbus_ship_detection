train_img_dir = "../data/train/"
test_img_dir = "../data/test/"
train_label_file = "../data/train_ship_segmentations.csv"
test_img_dir = "../data/test"
valid_ratio = 0.2
batch_size = 20
learning_rate = 0.0001
decay_rate = 0.999
saved_model = "./saved_model_2/model_1600_56126"
img_split_parts = 9
weight_ratio = 15 # must larger than 8


model_flag = "test"
continue_train_learning_rate = 0.00005
