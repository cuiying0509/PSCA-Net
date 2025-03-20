import torch


def do_test(test_dataloader, val_dataloader, model, num_test, threshold, device):
    if threshold is None:
        model.eval()
        err = torch.zeros(100).to(device)
        TP = torch.zeros(100).to(device)
        TN = torch.zeros(100).to(device)
        positive = 0
        negative = 0
        tmp = torch.linspace(0.01, 1, 100).reshape(1, 1, -1).to(device)
        # tmp = torch.linspace(1, 100, 100).reshape(1, 1, -1).to(device)
        with torch.no_grad():
            for vbatch, (va, vg, vH, vZ) in enumerate(val_dataloader):
                voutputs = model(va, vg, vH, vZ)
                pred = (torch.unsqueeze(voutputs, 2) > tmp).float()
                err = err + torch.abs(pred - torch.unsqueeze(va, 2))
        #         positive = positive + va.sum()
        #         negative = negative + (1-va).sum()
        #         TP = TP + pred.sum(axis=(0, 1)) - (pred - torch.unsqueeze(va, 2) > 0).float().sum(axis=(0, 1))
        #         TN = TN + (1-pred).sum(axis=(0, 1)) - (pred - torch.unsqueeze(va, 2) < 0).float().sum(axis=(0, 1))
        # score = 1 - 1/2 * (TP/positive + TN/negative)
        # threshold = torch.min(score, 0)[1] / 100
        err = err.sum(axis=(0, 1))
        threshold = torch.min(err, 0)[1] / 100
    # threshold = 1
    running_loss = 0.0
    TP_test = 0
    TN_test = 0
    positive_test = 0
    negative_test = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for batch, (a, g, H, Z) in enumerate(test_dataloader):
            outputs = model(a, g, H, Z)
            # pred = (outputs > threshold).float()
            # # err = err + torch.abs(pred - torch.unsqueeze(va, 2))
            # positive = positive + a.sum()
            # negative = negative + (1 - a).sum()
            # TP = TP + pred.sum() - (pred - a > 0).float().sum()
            # TN = TN + (1 - pred).sum() - (pred - a < 0).float().sum()
            # score = 1 - 1 / 2 * (TP / positive + TN / negative)
            loss = torch.sum(torch.abs((outputs > threshold).float() - a))
            running_loss += loss

    test_err = running_loss / (num_test * a.shape[1])
    # test_err = score

    a, g, H, Z = list(test_dataloader)[0]
    T = 0.0
    num_sample = a.shape[0]
    with torch.no_grad():
        for i in range(num_sample):
            starter.record()
            _ = model(torch.unsqueeze(a[i, :], 0),  torch.unsqueeze(g[i, :], 0),
                      torch.unsqueeze(H[i, :, :], 0), torch.unsqueeze(Z[i, :, :], 0))
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            T += curr_time

    test_time = T / num_sample
    return test_err, test_time, threshold
