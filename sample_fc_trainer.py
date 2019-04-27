import numpy as np
import torch
import torch.nn as nn
import functions as f
import archs.fully_connected as fc
import time
import matplotlib.pyplot as plt
import datasets

def train_fc(learning_rate = 1e-4,
             num_epochs = 50,
             decay_every = 50,
             gamma = 0.1,
             batch_size=256,
             num_batches=100,
             ):
    TRAIN_SET_SIZE = batch_size * num_batches

    file_name = "data/cifar-10-batches-py/data_batch_1"
    data = f.unpickle(file_name)

    ims = f.open_data_ims(50000)
    x,y = f.seperate_xy_ims(ims[0:TRAIN_SET_SIZE])

    # figure our device here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Using device:{}, cuda:{}, pytorch:{}".format(device, torch.version.cuda, torch.__version__))

    m = fc.FC_4(1536, 1536, 1536)
    if torch.cuda.is_available(): m = m.cuda()  # transfer model to cuda as needed
    lossFunction = nn.L1Loss()
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=gamma)

    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    x, y = x / 256.0, y / 256.0 # regularize

    # introduce a dataset class
    train_dataset = datasets.VectorizedDataset(x,y)
    train_loader = datasets.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0)

    # remember loss graph
    losses = np.zeros(num_epochs, dtype=np.float)

    print("Started training.")
    train_start_time = time.time()

    for epoch in range(num_epochs):
        if epoch != 0 and epoch % int(num_epochs/10) == 0:
            print("{}% done! loss:{}".format(epoch / num_epochs * 100.0, losses[epoch-1]))

        epoch_losses = 0.0

        # perform MINI-BATCH grad_descent here
        for i, data in enumerate(train_loader, 0):
            x_batch, y_batch = data

            y_pred = m.forward(x_batch)
            loss = lossFunction(y_pred, y_batch)
            epoch_losses += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses[epoch] = epoch_losses / num_batches


    print("Training took {} seconds!".format(time.time() - train_start_time))

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
