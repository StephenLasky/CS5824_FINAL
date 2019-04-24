import numpy as np
import torch
import torch.nn as nn
import functions as f
import archs.fully_connected as fc
import time

file_name = "data/cifar-10-batches-py/data_batch_1"
data = f.unpickle(file_name)

ims = f.open_data_ims(10000)
x,y = f.seperate_xy_ims(ims[0:10000])


LEARNING_RATE = 0.0001
NUM_EPOCHS = 100

m = fc.FC_2(1536, 500, 1536)
lossFunction = nn.L1Loss()
optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# regularize
x, y = x / 256.0, y / 256.0

print("Started training.")
train_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    y_pred = m.forward(x)
    loss = lossFunction(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training took {}".format(time.time() - train_start_time))

# print results
y_pred = m.forward(x)
x, y_pred = np.asarray(x.data), np.asarray(y_pred.data)
x, y_pred = x * 256.0, y_pred * 256.0
x, y_pred = x[0:250], y_pred[0:250]
fuse_xy_pred = f.combine_xy_ims(x,y_pred)
big_im = f.comb_ims(fuse_xy_pred,32,32)
f.show_im_std(big_im)