import torch
import torch.nn as nn
import datasets
import numpy as np
import time
from matplotlib import pyplot as plt

# train using only the top and bottom half of images
def train(x, y, m, x_test=None, y_test=None,
          learning_rate = 1e-4,
          decay_every = 50,
          gamma = 0.1,
          batch_size = 256,
          num_epochs = 100):
    # figure our device here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Using device:{}, cuda:{}, pytorch:{}".format(device, torch.version.cuda, torch.__version__))

    if torch.cuda.is_available(): m = m.cuda()  # transfer model to cuda as needed
    lossFunction = nn.L1Loss()
    # lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=gamma)

    # prep data
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    train_dataset = datasets.StdDataset(x, y)
    train_loader = datasets.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0)

    # remember loss graph
    train_losses = np.zeros(num_epochs, dtype=np.float)
    test_losses = np.zeros(num_epochs, dtype=np.float)
    test_size = x_test.shape[0]

    print("Started training.")
    train_start_time = time.time()

    for epoch in range(num_epochs):
        if num_epochs >= 10 and epoch != 0 and epoch % int(num_epochs/10) == 0:
            print("{}% done! lossTrain:{} lossTest:{}".format(epoch / num_epochs * 100.0,
                                                              train_losses[epoch-1],
                                                              test_losses[epoch-1]))

        epoch_losses = 0.0

        # perform MINI-BATCH grad_descent here
        z = 0
        for i, data in enumerate(train_loader, 0):
            x_batch, y_batch = data

            y_pred = m.forward(x_batch)
            loss = lossFunction(y_pred, y_batch)
            epoch_losses += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            z += 1


        train_losses[epoch] = epoch_losses / z
        y_pred_test = m.forward(x_test)
        test_losses[epoch] = lossFunction(y_pred_test, y_test)

    print("Training took {} seconds!".format(time.time() - train_start_time))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.show()

    return m