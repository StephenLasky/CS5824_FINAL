import numpy as np
import torch
import torch.nn as nn
import functions as f
import archs.fully_connected as fc
import time
import  matplotlib.pyplot as plt

file_name = "data/cifar-10-batches-py/data_batch_1"
data = f.unpickle(file_name)

ims = f.open_data_ims(10000)
x,y = f.seperate_xy_ims(ims)

# figure our device here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
print("Using device:{}".format(device))

LEARNING_RATE = 0.0001
NUM_EPOCHS = 300
DECAY_EVERY = 100

m = fc.FC_1(1536, 1536, 1536)
if torch.cuda.is_available(): m = m.cuda()  # transfer model to cuda as needed
lossFunction = nn.L1Loss()
optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# regularize
x, y = x / 256.0, y / 256.0

# remember loss graph
losses = np.zeros(NUM_EPOCHS, dtype=np.float)

print("Started training.")
train_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    if epoch != 0 and epoch % int(NUM_EPOCHS/10) == 0:
        print("{}% done!".format(epoch / NUM_EPOCHS * 100.0))

    # learning rate decay here
    if epoch != 0 and epoch % DECAY_EVERY == 0:
        LEARNING_RATE /= 2
        optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

    y_pred = m.forward(x)
    loss = lossFunction(y_pred, y)
    losses[epoch] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training took {}".format(time.time() - train_start_time))

# print loss graph
print(losses)
plt.plot(losses)
plt.show()

# print results
y_pred = m.forward(x)
x, y_pred = x.cpu().detach().numpy(), y_pred.detach().cpu().numpy()
# x, y_pred = np.asarray(x.data), np.asarray(y_pred.data)
x, y_pred = x * 256.0, y_pred * 256.0
x, y_pred = x[0:50], y_pred[0:50]
fuse_xy_pred = f.combine_xy_ims(x,y_pred)
big_im = f.comb_ims(fuse_xy_pred,32,32)
f.show_im_std(big_im)