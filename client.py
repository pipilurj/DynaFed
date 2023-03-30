import random
from tqdm import tqdm
from functools import partial
from collections import OrderedDict
import torch
import torch.optim as optim
#from torchcontrib.optim import SWA
import torch.nn as nn
import numpy as np 
from utils import *
import models as model_utils
from sklearn.linear_model import LogisticRegression

import os

# from gmm_torch.gmm import GaussianMixture
from math import sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class Device(object):
  def __init__(self, loader):
    
    self.loader = loader

  def evaluate(self, loader=None):
    return eval_op(self.model, self.loader if not loader else loader)

  def save_model(self, path=None, name=None, verbose=True):
    if name:
      torch.save(self.model.state_dict(), path+name)
      if verbose: print("Saved model to", path+name)

  def load_model(self, path=None, name=None, verbose=True):
    if name:
      self.model.load_state_dict(torch.load(path+name))
      if verbose: print("Loaded model from", path+name)
  
class Client(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10, images_train=None, labels_train=None, eta=0.5, dataset = 'cifar10'):
    super().__init__(loader)
    self.id = idnum
    print(f"dataset client {dataset}")
    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes , dataset = dataset)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())
    self.images_train, self.labels_train = images_train, labels_train
    self.eta = eta

    
  def synchronize_with_server(self, server):
    server_state = server.model_dict[self.model_name].state_dict()
    self.model.load_state_dict(server_state, strict=False)
# server_state,server.parameter_dict['resnet8'], self.model.state_dict()
    
  def compute_weight_update(self, epochs=1, loader=None, lambda_fedprox=0.0, print_train_loss=False,  hp=None):
    clip_bound, privacy_sigma = None, None
    if hp is not None:
      clip_bound, privacy_sigma = hp.get("clip_bound", None), hp.get("privacy_sigma", None)
    if privacy_sigma is not None:
      train_stats = train_op_private(self.model, self.loader if not loader else loader, self.optimizer, epochs, lambda_fedprox=lambda_fedprox, print_train_loss=print_train_loss, clip_bound=clip_bound, privacy_sigma=privacy_sigma)
    else:
      train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs, lambda_fedprox=lambda_fedprox, print_train_loss=print_train_loss)
    return train_stats

  def compute_weight_update_datadistill(self, epochs=1, loader=None, lambda_fedprox=0.0, current_round=0, start_round=0):
    print(f"current round {current_round}, start round {start_round}")
    if self.images_train is not None and self.labels_train is not None:
      train_stats = train_op_datadistill(self.model, self.loader if not loader else loader, self.optimizer, epochs, self.images_train, self.labels_train, eta=self.eta, current_round=current_round, start_round=start_round)
    else:
      train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs, lambda_fedprox=lambda_fedprox)
    return train_stats

  def compute_weight_update_datadistill_soft(self, epochs=1, loader=None, lambda_fedprox=0.0, current_round=0, start_round=0, dsa=True, args=None):
    # print(f"soft distill, current round {current_round}, start round {start_round}")
    if self.images_train is not None and self.labels_train is not None:
      train_stats = train_op_datadistill_soft(self.model, self.loader if not loader else loader, self.optimizer, epochs, self.images_train, self.labels_train, eta=self.eta, current_round=current_round, start_round=start_round, dsa=dsa, args=args)
    else:
      train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs, lambda_fedprox=lambda_fedprox)
    return train_stats

  def compute_weight_update_datadistill_later(self, epochs=1, loader=None, lambda_fedprox=0.0, finetune_lr=1e-3, finetune_epoch=1, current_round=0, start_round=0, dsa=None, args=None):
    if self.images_train is not None and self.labels_train is not None:
      train_stats = train_op_datadistill_later(self.model, self.loader if not loader else loader, self.optimizer, epochs, self.images_train, self.labels_train, finetune_epoch=finetune_epoch, finetune_lr=finetune_lr, current_round=current_round, start_round=start_round, dsa=dsa, args=args)
    else:
      train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs, lambda_fedprox=lambda_fedprox)
    return train_stats


  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_
  
  def predict_logit_eval(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = self.model(x)

    return y_

