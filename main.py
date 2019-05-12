import numpy as np
import torch
import torchvision
import functions as f
import sample_fc_trainer as fct
import var_size_FC_trainer as vfct
from multiprocessing import Process
import archs.cnn as cnns
import basic_CNN_trainer

# TODO: UNCOMMENT BELOW TO CONTINUE WHAT WE WERE WORKING ON!!!
# vfct.train_fc(num_epochs=1,decay_every=4, batch_size= 290000,
#               x_train_filename="data/x3y1_10k/x_train.npy",
#               y_train_filename="data/x3y1_10k/y_train.npy",
#               num_x_rows=3,
#               num_y_rows=1,
#               hidden_width=256,
#               shuffle_data=False,
#               display_info=False)

DATA_FOLDER = "./data/custom/"


train_size = 47 * 1024
test_size = 1024
size = train_size + test_size

# x, y = f.load_basic_dataset_chw(size)
# x, y = f.regularize_data(x), f.regularize_data(y)
#
# x_train = x[0:train_size]
# y_train = y[0:train_size]
# x_test = x[train_size:]
# y_test = y[train_size:]
# x_test = x[0:train_size]
# y_test = y[0:train_size]

x_train = torch.load(DATA_FOLDER+"x_train.pt")
y_train = torch.load(DATA_FOLDER+"y_train.pt")
x_test = torch.load(DATA_FOLDER+"x_test.pt")
y_test = torch.load(DATA_FOLDER+"y_test.pt")

m = cnns.CNN_7()
m = basic_CNN_trainer.train(x_train, y_train,  m, x_test=x_test, y_test=y_test,
                            num_epochs=10,
                            decay_every=10,
                            learning_rate=1.5e-3,
                            batch_size= 1024)



# get, transform, and display test results
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
x_test = torch.tensor(x_test, dtype=torch.float)
y_pred = m.forward(x_test)
x_test, y_pred = x_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
x_test_std = f.chw_ims_to_std(x_test)
y_pred_std = f.chw_ims_to_std(y_pred)

# y_pred_std = f.regularize_0_1(y_pred_std)
combined = f.combine_xy_ims_std(x_test_std, y_pred_std)
big_im = f.comb_ims_std(combined)
# big_im = (big_im-np.min(big_im))/(np.max(big_im)-np.min(big_im)) * 255.0
big_im *= 255.0

print(np.mean(big_im), np.min(big_im), np.max(big_im))
f.show_im_std(big_im)
