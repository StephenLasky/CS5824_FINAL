import numpy as np
import torch
import torch.nn as nn
import functions as f
import archs.fully_connected as fc
import time

file_name = "data/cifar-10-batches-py/data_batch_1"
data = f.unpickle(file_name)

ims = f.open_data_ims(10000)
x,y = f.seperate_xy_ims(ims[0:250])


LEARNING_RATE = 0.01
NUM_EPOCHS = 20
DECAY_EVERY = 2

m = fc.FC_2(1536, 768, 1536)
lossFunction = nn.L1Loss()
optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# regularize
x, y = x / 256.0, y / 256.0

print("Started training.")
train_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # learning rate decay here
    if epoch != 0 and epoch % DECAY_EVERY == 0:
        LEARNING_RATE /= 2
        optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

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
x, y_pred = x[0:50], y_pred[0:50]
fuse_xy_pred = f.combine_xy_ims(x,y_pred)
big_im = f.comb_ims(fuse_xy_pred,32,32)
f.show_im_std(big_im)