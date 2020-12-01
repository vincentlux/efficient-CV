import os
import json
import numpy as np
from PIL import Image

import time
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

import mlflow

from models.resnet import resnet10, resnet18, resnet50, resnet101
from params import args
import helper
from tqdm import tqdm

import logging
from torch.nn.utils import prune
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_MAP = {
    'resnet10': resnet10,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
}

def train(model, train_loader, epoch, criterion, optimizer, warmup_scheduler):
    correct = 0
    avg_loss = 0.
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if epoch <= args.warmup_steps:
            warmup_scheduler.step()
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

def valid(model, eval_loader, criterion, task):
    model.eval()
    eval_loss = 0
    correct = 0
    total_secs = []
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(args.device), target.to(args.device)
            start = time.time()

            if task == 'fp16':
                data = data.half()

            output = model(data)
            total_secs.append(time.time() - start)
            eval_loss += criterion(output, target).sum().item()  # sum up batch loss
            _, preds = output.max(1)
            correct += preds.eq(target).sum().item()

    eval_loss /= len(eval_loader.dataset)
    eval_accu = correct / len(eval_loader.dataset)
    logger.info('\nEval set: Average loss: {:.5f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        eval_loss, correct, len(eval_loader.dataset),
        100. * correct / len(eval_loader.dataset)))
    latency = sum(total_secs) * 1000 / len(total_secs)
    return eval_loss, eval_accu, latency


def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

def load_model(model_name, state_dict_path=None):
    model = MODEL_MAP[model_name]()

    if state_dict_path:
        state = torch.load(state_dict_path)
        model.load_state_dict(state['state_dict'])
        logger.info(f'Loaded the model of epoch {state["epoch"]} from {state_dict_path}')
    return model

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
        
        dc = DeviceCheck()
        criterion = nn.CrossEntropyLoss()

    if args.do_train:
        # set device
        device_name, device_ids = dc.get_device(n_gpu=args.n_gpu)
        device_name = '{}:{}'.format(device_name, device_ids[0]) if len(device_ids) == 1 else device_name
        args.device = torch.device(device_name)
        logger.info(f'device: {args.device}')

        # Initialize the model for this run
        if args.student_model_name:
            logger.info('Start distillation training...')
            logger.info('Loading trained teacher weights...')
            model = load_model(args.model_name, args.test_model_path)
            model.eval()

            s_model = load_model(args.student_model_name)
            s_model.to(args.device)
        else:
            model = load_model(args.model_name)

        model.to(args.device)

        train_loader = helper.get_dataloader('train')
        eval_loader = helper.get_dataloader('eval')

        optimizer = helper.init_optimizer(model)

        iter_per_epoch = len(train_loader)
        warmup_scheduler = helper.WarmUpLR(optimizer, iter_per_epoch * args.warmup_steps)
        scheduler = helper.init_scheduler(optimizer)

        # valid once before training
        logger.info('validation before training')
        valid_loss, valid_accu, _ = valid(model, eval_loader, criterion, None)
        mlflow.log_metric("zero-shot-accuracy/valid" , valid_accu, 0)

        max_accu = 0.0
        for epoch in tqdm(range(1, args.num_epochs + 1)):
            train_accu = train(model, train_loader, epoch, criterion, optimizer, warmup_scheduler)
            valid_loss, valid_accu, _ = valid(model, eval_loader, criterion, None)

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
        #TODO add 
        # loading model [done]
        # time it [done]
        # quantization [done]
        # fp16 [done]
        # distillation
        # pruning

        # Assign tasks
        tasks = args.benchmarks.split(',')
        logger.info('Tasks: {}'.format(tasks))

        # Look for gpu devices
        assert args.n_gpu <= 1, 'multi gpu inference not supported yet'
        gpu_device_name, gpu_device_ids = dc.get_device(n_gpu=args.n_gpu)
        gpu_device_name = '{}:{}'.format(gpu_device_name, gpu_device_ids[0]) if len(gpu_device_ids) == 1 else gpu_device_name
        device_names = [gpu_device_name, 'cpu']
        pruning_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thres in pruning_threshold:
            # Loop through both cpu and gpu
            for device in device_names:
                for task in tasks:
                    if thres > 0 and task != 'pruning':
                        continue
                    
                    if task == 'baseline':
                        model = load_model(args.model_name, args.test_model_path)
                        pass

                    elif task == 'quantization':
                        if device != 'cpu':
                            logger.info('quantization only works in cpu env, continue...')
                            continue
                        model = load_model(args.model_name, args.test_model_path)
                        model = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear}, dtype=torch.qint8)
                    
                    elif task == 'fp16':
                        if device == 'cpu':
                            logger.info('fp16 only works in gpu env, continue...')
                            continue
                        model = load_model(args.model_name, args.test_model_path)
                        model.half()

                    elif task == 'distillation':
                        #TODO
                        # override model with distilled model and reload weights
                        distill_model = load_model(args.student_model_name, args.student_test_model_path)
                        raise NotImplementedError
                    
                    elif task == 'pruning':
                        if device == 'cpu':
                            continue

                        model = load_model(args.model_name, args.test_model_path)
                        params = [(module, 'weight') for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)]
                        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=thres)

                    else:
                        logger.warn('{} not handled, pass'.format(task))
                        continue

                    # send model to device
                    args.device = torch.device(device)
                    logger.info(f'Evaluating {task} on device: {args.device}')
                    model.to(args.device)
                    eval_loader = helper.get_dataloader('eval')
                    try:
                        valid_loss, valid_accu, latency = valid(model, eval_loader, criterion, task)
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()

                    size_in_mb, num_params = helper.get_size_of_model(model)
                    
                    # Logging
                    suffix = '{}_{}_{}'.format(task, 'gpu' if 'cuda' in device else 'cpu', str(thres))
                    res_dict = {
                        f'accuracy/{suffix}': valid_accu,
                        f'latency_ms/{suffix}': latency,
                        f'size_mb/{suffix}': size_in_mb,
                        f'params_million/{suffix}': num_params
                        }

                    logger.info(res_dict)
                    mlflow.log_metrics(res_dict)
                    helper.save_logs(args.output_dir, res_dict)
                    


if __name__ == '__main__':
    main()
    
