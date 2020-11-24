import os
import json
import numpy as np
from PIL import Image

import torch
import torchvision  
from torchvision import transforms 
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from soco_device import DeviceCheck
from datetime import datetime

import torchvision.models as models
import mlflow

from params import args
import helper
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, train_loader, epoch, criterion, optimizer):
    correct = 0
    avg_loss = 0.
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # import pdb; pdb.set_trace()
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = criterion(output, target)
        train_loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss += train_loss.item()
        if batch_idx % 50 == 49:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_loss/50))
            avg_loss = 0.

    train_accu = correct / len(train_loader.dataset)
    logger.info('Train accuracy: {}/{}({:.3f}%)'.format(
        correct, len(train_loader.dataset),
        100. * train_accu))
    return train_accu

def valid(model, eval_loader, criterion):
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            eval_loss += criterion(output, target).sum().item()  # sum up batch loss
            _, preds = output.max(1)
            correct += preds.eq(target).sum().item()
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    eval_loss /= len(eval_loader.dataset)
    eval_accu = correct / len(eval_loader.dataset)
    logger.info('\nEval set: Average loss: {:.5f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        eval_loss, correct, len(eval_loader.dataset),
        100. * correct / len(eval_loader.dataset)))
    return eval_loss, eval_accu


def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

def main():
    if args.do_train or args.do_eval:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%dT%H-%M-%S")
        print("date and time =", dt_string)
        args.output_dir = 'snap/' if not args.output_dir else args.output_dir
        args.output_dir = os.path.join(args.output_dir, dt_string)
        os.makedirs(args.output_dir)

        dict_config = vars(args)
        json.dump(dict_config, open(os.path.join(args.output_dir, 'dict_config.json'), 'w'), indent=2)
        mlflow.set_experiment(__file__.split('/')[-1].split('.')[0])
        mlflow.log_params(dict_config)

    # set device
    dc = DeviceCheck()
    device_name, device_ids = dc.get_device(n_gpu=args.n_gpu)
    device_name = '{}:{}'.format(device_name, device_ids[0]) if len(device_ids) == 1 else device_name
    args.device = torch.device(device_name)
    logger.info(f'device: {args.device}')


    # Initialize the model for this run
    model = models.resnet18(pretrained=args.use_pretrained, num_classes=100)
    model.to(args.device)
    print(model)

    if args.init == 'xavier':
        logger.info('using xavier init')
        model.apply(init_xavier)
    logger.info(model)
    
    criterion = nn.CrossEntropyLoss()
    
    if args.do_train:
        train_loader = helper.get_dataloader('train')
        eval_loader = helper.get_dataloader('eval')

        optimizer = helper.init_optimizer(model)

        scheduler = helper.init_scheduler(optimizer)

        # valid once before training
        logger.info('validation before training')
        valid_loss, valid_accu = valid(model, eval_loader, criterion)
        mlflow.log_metric("zero-shot-accuracy/valid" , valid_accu, 0)

        max_accu = 0.0
        for epoch in tqdm(range(1, args.num_epochs + 1)):
            train_accu = train(model, train_loader, epoch, criterion, optimizer)
            valid_loss, valid_accu = valid(model, eval_loader, criterion)

            mlflow.log_metric("accuracy/train" , train_accu, epoch)
            mlflow.log_metric("accuracy/valid" , valid_accu, epoch)
            mlflow.log_metric("loss/valid" , valid_loss, epoch)
            if args.scheduler in ['steplr', 'multistep']:
                scheduler.step()
            elif args.scheduler == 'plateau':
                scheduler.step(valid_loss)
            else:
                raise NotImplementedError
            
            if valid_accu > max_accu:
                state = {
                        'epoch': epoch,
                        'eval_accuracy': valid_accu,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }
                current_best_model_name = os.path.join(args.output_dir, args.best_pt_name)
                torch.save(state, current_best_model_name)
                logger.info(f'\nsaved best model to {current_best_model_name} with eval accuracy {valid_accu} from epoch {epoch}')
                max_accu = valid_accu

    if args.do_eval:
        #TODO add loading model, quantization, pruning, etc
        eval_loader = helper.get_dataloader('eval')
        valid_loss, valid_accu = valid(model, eval_loader, criterion)
        
        mlflow.log_metric("accuracy/valid" , valid_accu, epoch)


if __name__ == '__main__':
    main()
    