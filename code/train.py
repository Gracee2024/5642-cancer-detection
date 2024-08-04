import sys
import warnings
import torch
import time
import random
from torch.nn import CrossEntropyLoss
import os
from torch.utils.data import DataLoader
import numpy as np
from utils import print_dct
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


try:
    sys.path.insert(0, '.')
    from model.base import Model
    from datasets.base import Dataset
    from evaluation import evaluate
    from optimizer import Optimizer
    from scheduler import Scheduler, print_scheduler
    from utils import parse, get_logger
finally:
    pass

warnings.filterwarnings('ignore')


def train():
    args = parse()
    args_str = print_dct(args)

    torch.manual_seed(args['train']['seed'])
    random.seed(args['train']['seed'])

    epochs = args['train']['epochs']
    bs = args['dataset']['batch_size']
    device_str = args['train']['device'] if torch.cuda.is_available() else 'cpu'
    # se = args['train']['save_every']

    optimizer_name = args['optimizer']['name']
    optimizer_params = args['optimizer']['parameters']

    model = Model(args['model'])
    model.to(device_str)

    optimizer = Optimizer(optimizer_name, optimizer_params, model.parameters())
    # scheduler = Scheduler(optimizer, args['scheduler'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率调整

    loss_fn = CrossEntropyLoss()

    if 'checkpoint' in args['train'].keys():
        model.load_state_dict(torch.load(args['train']['checkpoint']))

    log_path = f"{args['train']['log_path']}{args['model']['name'].lower()}/"
    os.makedirs(log_path, exist_ok=True)
    logger = get_logger(log_path + 'train.log')

    save_path = f"{args['train']['save_path']}{args['model']['name'].lower()}/"
    save_name = f'{args["model"]["name"]}.pth'
    os.makedirs(save_path, exist_ok=True)

    logger.info(f'\n{args_str}')
    logger.info(f'\n{str(model)}')
    logger.info(f'\n{str(optimizer)}')
    logger.info(f'\n{str(loss_fn)}')

    train_dataset, valid_dataset = Dataset(args['dataset'])

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=bs,
        shuffle=True
    )

    init_time = time.time()
    best_loss = 99999999.0

    epochs_npz = np.array([ep for ep in range(epochs)])
    train_loss_npz = []
    valid_loss_npz = []
    train_acc_npz = []
    valid_acc_npz = []

    for epoch in tqdm(range(epochs)):
        start = time.time()
        model.train()

        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        for batch, (imgs, labels) in enumerate(train_data_loader):
            # forward
            imgs, labels = imgs.to(device_str), labels.to(device_str)
            preds = model(imgs)
            loss = loss_fn(preds, labels)

            y_actual = labels.data.cpu().numpy()
            y_pred = preds.argmax(1).detach().cpu().numpy()
            train_acc += accuracy_score(y_actual, y_pred)
            # train_acc_npz.append(accuracy_score(y_actual, y_pred))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for batch, (imgs, labels) in enumerate(valid_data_loader):
                imgs, labels = imgs.to(device_str), labels.to(device_str)
                preds = model(imgs)
                loss = loss_fn(preds, labels)

                y_actual = labels.data.cpu().numpy()
                y_pred = preds.argmax(1).detach().cpu().numpy()
                valid_acc += accuracy_score(y_actual, y_pred)
                valid_loss += loss.item()

        train_loss /= len(train_data_loader)
        valid_loss /= len(valid_data_loader)
        train_acc /= len(train_data_loader)
        valid_acc /= len(valid_data_loader)

        train_loss_npz.append(train_loss)
        valid_loss_npz.append(valid_loss)
        train_acc_npz.append(train_acc)
        valid_acc_npz.append(valid_acc)
        cond1 = valid_loss < best_loss

        if cond1:
            best_loss = valid_loss
            state_dict = model.state_dict()
            torch.save(state_dict, save_path + save_name)
        scheduler.step()

        end = time.time()

        print_str = f'{device_str} '
        print_str += f'epoch: {epoch + 1}/{epochs} '
        print_str += f'train_loss: {train_loss:.4f} '
        print_str += f'valid_loss: {valid_loss:.4f} '
        print_str += f'train_acc: {train_acc:.4f} '
        print_str += f'valid_acc: {valid_acc:.4f} '
        print_str += f'epoch_time: {(end - start):.3f} sec'
        logger.info(print_str)

    last_time = time.time()
    print(save_path + save_name)
    model = Model(args['model']).to(device_str)
    model.load_state_dict(torch.load(save_path + save_name))

    train_metrics = evaluate(model, train_data_loader, device_str)
    valid_metrics = evaluate(model, valid_data_loader, device_str)

    print_str = '\n'

    for name, val in train_metrics:
        print_str += f'train_{name}: {val:.3f} '

    for name, val in valid_metrics:
        print_str += f'valid_{name}: {val:.3f} '

    print_str += f'total_time: {(last_time - init_time):.3f} sec'
    logger.info(print_str)

    np.savez_compressed(
        log_path + 'losses.npz',
        epochs_npz,
        train_loss_npz,
        valid_loss_npz,
        train_acc_npz,
        valid_acc_npz
    )


if __name__ == '__main__':
    train()
