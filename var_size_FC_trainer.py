import functions as f
import numpy as np
import torch
import torch.nn as nn
import datasets
import archs.fully_connected as fc
import time

def train_fc(learning_rate = 1e-4,
             num_epochs = 50,
             decay_every = 50,
             gamma = 0.1,
             batch_size=1024 * 20,
             num_batches=20,
             display_info = True,
             hidden_width = 512,
             num_x_rows = 3,
             num_y_rows = 1,
             x_train_filename = "data/custom/x_train.npy",
             y_train_filename =  "data/custom/y_train.npy",
             shuffle_data = True
             ):
    TRAIN_SET_SIZE = batch_size * num_batches
    ROW_WIDTH = 32
    COL_HEIGHT = 32
    COLOR_CHANNELS = 3

    try:
        print("Attempting to load x, y data from disk.")
        x = np.load(x_train_filename)
        y = np.load(y_train_filename)
    except:
        # create the dataset that we will be training on
        print("x,y data disk load failure. Generating from scratch...")
        ims = f.open_data_ims(50000)
        x, y = f.generate_custom_data(num_x_rows, num_y_rows, ims[0:TRAIN_SET_SIZE], 32, 32)
        np.save(x_train_filename, x)
        np.save(y_train_filename, y)
    assert x.shape[0] == y.shape[0]     # ensure x,y are of equal lengths

    # recombined = f.combine_xy_ims(x,y)
    # big_im = f.comb_ims(recombined, im_width=32, im_height=num_x_rows+num_y_rows)
    # f.show_im_std(big_im)

    # now begin setting up the model
    input_width = ROW_WIDTH * num_x_rows * COLOR_CHANNELS
    output_width = ROW_WIDTH * num_y_rows * COLOR_CHANNELS

    # set up devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Using device:{}, cuda:{}, pytorch:{}".format(device, torch.version.cuda, torch.__version__))

    # set up model, optimizer, scheduler
    m = fc.FC_2(input_width, hidden_width, output_width)
    if torch.cuda.is_available(): m = m.cuda()  # transfer model to cuda as needed
    lossFunction = nn.L1Loss()
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=gamma)

    # ensure x, y are of correct form
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    x, y = x / 256.0, y / 256.0  # regularize

    print("Data set size:{}".format(x.shape[0]))

    train_dataset = datasets.VectorizedDataset(x, y)
    train_loader = datasets.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle_data,
                                       num_workers=0)

    # remember loss graph
    losses = np.zeros(num_epochs, dtype=np.float)

    print("Started training.")
    train_start_time = time.time()

    for epoch in range(num_epochs):
        if num_epochs >= 10 and epoch != 0 and epoch % int(num_epochs / 10) == 0:
            print("{}% done! loss:{}".format(epoch / num_epochs * 100.0, losses[epoch - 1]))

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

    # our model is now trained. now let us attempt to reconstruct samples from our train set
    VAL_SET_SIZE = 100
    ims = f.open_data_ims(10000)[0:VAL_SET_SIZE]

    # lets try to start predicting from halfway down
    # ims = np.reshape(ims,(VAL_SET_SIZE, ROW_WIDTH, COL_HEIGHT, COLOR_CHANNELS)) # to STD form
    YS = 16
    # ims = torch.tensor(ims, dtype=torch.float)
    for im in range(0,VAL_SET_SIZE):
        for i in range(YS, COL_HEIGHT + 1 - num_y_rows):
            ys = i                      # y row start
            ye = ys + num_y_rows        # y row end (noninclusive)
            xs = ys - num_x_rows        # x row start
            xe = ys                     # x row end (noninclusive)

            x = ims[im]
            x = f.extract_vec_rows(xs,xe,32,x)
            x = torch.tensor(x, dtype=torch.float)

            y_pred = m.forward(x)
            y_pred = y_pred.cpu().detach().numpy()
            ims[im] = f.paste_over_rows(ys,ye,32,ims[im],y_pred)


    ims = np.reshape(ims, (VAL_SET_SIZE, ROW_WIDTH * COL_HEIGHT * COLOR_CHANNELS))
    big_im = f.comb_ims(ims, 32, 32)

    if display_info:
        f.show_im_std(big_im)
    # all should now be predicted. lets display them!
