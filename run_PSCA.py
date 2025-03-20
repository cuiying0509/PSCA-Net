import argparse
import datetime
import logging
import os
import sys
import time
import pickle
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from PSCANet import PSCANet, PSCANetMAP
from data import data_generator, data_loader
from test_PSCA import do_test
from train_PSCA import do_train

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

parser = argparse.ArgumentParser(description='PSCANet experiment')
parser.add_argument('--N', default=1000, type=int, help='user number')
parser.add_argument('--L', default=40, type=int, help='pilot length')
parser.add_argument('--M', default=256, type=int, help='antenna number')
parser.add_argument('--R', default=200, type=float, help='area radius')
parser.add_argument('--alpha', default=2.5, type=float, help='path loss exponent')
parser.add_argument('--sigma2', default=1e-9, type=float, help='variance of AWGN')
parser.add_argument('--pathloss', default="K", type=str, help='K for known, U for unknown')
parser.add_argument('--c', default="[-2.9444, 0]", type=str, help='MVB coefficient')
parser.add_argument('--mode', default="PSCANet", help='PSCANet, PSCA')
parser.add_argument('--pilot', default="random", help="ramdom, pretrain, zc")
parser.add_argument('--learning_rate', default=1e-2, type=float, help='learning rate')
parser.add_argument('--save_pilot', default="None", help='save pilot matrix: None, Yes')
parser.add_argument('--train_S', default="No", help='train pilot: Yes, No')
parser.add_argument('--train_gamma', default="Yes", help='train step size: Yes, No')
parser.add_argument('--train_c', default=None, help='train MVB coefficient: None, Yes, No')
parser.add_argument('--num_layer', default=30, type=int, help='network layer number')
parser.add_argument('--device', default="cuda:0", type=str, help='device')
parser.add_argument('--batch_size_train', default=64, type=int, help='train batch size')
parser.add_argument('--batch_size_val', default=2000, type=int, help='val batch size')
parser.add_argument('--batch_size_test', default=2000, type=int, help='test batch size')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--num_train', default=6000, type=int, help='number of train date')
parser.add_argument('--num_val', default=2000, type=int, help='number of validation date')
parser.add_argument('--num_test', default=2000, type=int, help='number of test date')
parser.add_argument('--n_jobs', default=-1, type=int, help='jobs for data generation')
parser.add_argument('--config', help="configuration file", type=str, default="configs/default.yml")
parser.add_argument('--threshold', help="predefined threshold", type=float, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    # process argparse & yaml
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    args = argparse.Namespace(**opt)

    N = args.N
    L = args.L
    M = args.M
    R = args.R
    alpha = args.alpha
    sigma2 = 1e-9
    pathloss = args.pathloss
    c = args.c
    train_S = args.train_S
    train_gamma = args.train_gamma
    train_c = args.train_c
    mode = args.mode
    pilot = args.pilot
    save_pilot = args.save_pilot
    num_layer = args.num_layer
    num_train = args.num_train
    num_val = args.num_val
    num_test = args.num_test
    n_jobs = args.n_jobs
    epochs = args.epochs
    batch_size_train = args.batch_size_train
    batch_size_val = args.batch_size_val
    batch_size_test = args.batch_size_test
    device = args.device

    data_info = "_N" + str(N) + "_L" + str(L) + "_M" + str(M) + "_c" + str(c)
    experiment_info = data_info + "_layer" + str(num_layer) + "_trainC" + str(args.train_c) + "_trainS" + str(
        args.train_S) + "_trainGamma" + str(args.train_gamma)
    args.exp_name = mode + "_" + experiment_info + "_" + datetime.datetime.now().strftime("%m%d%H%M%S")

    # make experiment directory
    if not os.path.exists(os.path.join("experiment", args.exp_name)):
        os.makedirs(os.path.join("experiment", args.exp_name))

    # make logging
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m-%d %I:%M:%S %p")

    fh = logging.FileHandler(os.path.join("experiment", args.exp_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    # generate data
    data_path_prefix = "data/data" + data_info + "_"
    if not os.path.isfile(data_path_prefix + "train" + ".pkl.bz2"):
        data_generator.Signal(N, L, M, R, alpha, sigma2, c, num_train, num_val, num_test, n_jobs,
                              data_path_prefix).train_val_test_generator()

    # make tensorboard file
    writer = SummaryWriter("experiment/%s/runs/%s" %
                           (args.exp_name, time.strftime("%m%d%H%M%S", time.localtime())))

    # load data
    data_train_path = data_path_prefix + "train" + ".pkl.bz2"
    data_val_path = data_path_prefix + "val" + ".pkl.bz2"
    data_test_path = data_path_prefix + "test" + ".pkl.bz2"

    train_dataloader = data_loader.make_data_loader(data_train_path, batch_size_train, device, shuffle=True)
    val_dataloader = data_loader.make_data_loader(data_val_path, batch_size_val, device, shuffle=False)
    test_dataloader = data_loader.make_data_loader(data_test_path, batch_size_val, device, shuffle=False)

    # build PSCANet
    if train_c is None:
        model = PSCANet(N, L, M, sigma2, pilot, train_S, train_gamma, pathloss, num_layer).to(device)
        best_model = PSCANet(N, L, M, sigma2, pilot, train_S, train_gamma, pathloss, num_layer).to(device)
    else:
        model = PSCANetMAP(N, L, M, sigma2, pilot, train_S, train_gamma, pathloss, c, train_c, num_layer).to(device)
        best_model = PSCANetMAP(N, L, M, sigma2, pilot, train_S, train_gamma, pathloss, c, train_c, num_layer).to(device)

    if mode == "PSCANet":
        # train
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


        do_train(train_dataloader, val_dataloader, model, pathloss, optimizer, writer, epochs,
                 os.path.join("experiment", args.exp_name))

        best_model.load_state_dict(
            torch.load(os.path.join("experiment", args.exp_name, 'model_best.pt'), map_location=device))

        if save_pilot == "Yes":
            with open("pilot/" + mode + "_" + pilot + "_r_N" + str(N) + "_L" + str(L) + ".pkl.bz2", 'wb') as f:
                pickle.dump(best_model.S_r.detach().cpu(), f)

            with open("pilot/" + mode + "_" + pilot + "_i_N" + str(N) + "_L" + str(L) + ".pkl.bz2", 'wb') as f:
                pickle.dump(best_model.S_i.detach().cpu(), f)
        # test
        test_err, test_time, threshold = do_test(test_dataloader, val_dataloader, best_model, num_test, args.threshold,
                                                 device)
        args.threshold = threshold.item()
    else:
        if save_pilot == "Yes":
            with open("pilot/" + mode + "_" + pilot + "_r_N" + str(N) + "_L" + str(L) + ".pkl.bz2", 'wb') as f:
                pickle.dump(best_model.S_r.detach().cpu(), f)

            with open("pilot/" + mode + "_" + pilot + "_i_N" + str(N) + "_L" + str(L) + ".pkl.bz2", 'wb') as f:
                pickle.dump(best_model.S_i.detach().cpu(), f)

        test_err, test_time, threshold = do_test(test_dataloader, val_dataloader, model, num_test, args.threshold,
                                                 device)
        # args.threshold = threshold.item()

    # print experiment result
    logging.info('Test error: {}'.format(test_err))
    logging.info('Time cost: {} ms'.format(test_time))

    # save yaml
    with open(os.path.join("experiment", args.exp_name, "config.yml"), "w") as f:
        yaml.dump(args, f)
