# Initializing in a separate cell, so we can easily add more epochs to the same run
import gc
import logging
import os

# import bcolors
import torch
import torch.nn as nn
from early_stopping import EarlyStopping


# torch.autograd.set_detect_anomaly(True)

def do_train(train_dataloader, val_dataloader, model, pathloss, optimizer, writer, epochs, model_path):
    epoch_number = 0
    best_vloss = 100_000_000.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, )
    es = EarlyStopping()
    if pathloss == "K":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        gc.collect()
        torch.cuda.empty_cache()
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for batch, (a, g, H, Z) in enumerate(train_dataloader):

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(a, g, H, Z).clamp(1e-3, 1 - 1e-3)

            # Compute the loss and its gradients
            if pathloss == "K":
                loss = loss_fn(outputs.view(-1, ), a.view(-1, ))
            else:
                loss = loss_fn(outputs.view(-1, ), (g * a).view(-1, ))
            # loss = loss_fn(outputs.view(-1, ), a.view(-1, ))
            # with torch.autograd.detect_anomaly():
            loss.backward()
            # for name, para in model.named_parameters():
            #     print('-->name:', name, '-->grad_requires:', para.requires_grad, ' -->grad_value_TOTAL:', torch.max(para.grad ** 2))

            optimizer.step()
            if hasattr(model, "gamma"):
                model.gamma.data.clamp(1e-2, 1 - 1e-2)
            # gather data and report
            running_loss += loss.item()
            if batch % 10 == (10 - 1):
                last_loss = running_loss / 10  # loss per batch
                print(' batch {} loss: {}'.format(batch + 1, last_loss / a.shape[1]))
                tb_x = epoch * len(train_dataloader) + batch + 1
                writer.add_scalar('Loss/train', last_loss / a.shape[1], tb_x)
                running_loss = 0.

        running_vloss = 0.
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for vbatch, (va, vg, vH, vZ) in enumerate(val_dataloader):
                voutputs = model(va, vg, vH, vZ).clamp(1e-3, 1 - 1e-3)
                if pathloss == "K":
                    vloss = loss_fn(voutputs.view(-1, ), va.view(-1, ))
                else:
                    vloss = loss_fn(voutputs.view(-1, ), (vg * va).view(-1, ))
                # vloss = loss_fn(voutputs.view(-1, ), va.view(-1, ))
                running_vloss += vloss

        avg_vloss = running_vloss / (vbatch + 1)
        print('LOSS train {} valid {}'.format(last_loss / a.shape[1], avg_vloss / a.shape[1]))

        if es.step(avg_vloss):
            break  # early stop criterion is met, we can stop now

        scheduler.step(avg_vloss)

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Validation Loss',
                           {'Validation': avg_vloss / a.shape[1]},
                           epoch_number + 1)
        writer.flush()

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # save_path = os.path.join(model_path + '/model_{}'.format(epoch_number))
            save_path = os.path.join(model_path, 'model_best.pt')
            torch.save(model.state_dict(), save_path)

        epoch_number += 1

    logging.info('Training stops at epoch {}'.format(epoch_number))
