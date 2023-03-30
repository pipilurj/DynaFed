import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import tqdm
from utils import kd_loss, DiffAugment
import wandb
import copy
from torch.utils.data import Dataset
from copy import deepcopy
import time
import random
from reparam_module import ReparamModule
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
mean_dataset={
    "cifar10":  [0.4914, 0.4822, 0.4465],
    "mnist":  [0.1307],
    "fmnist":  [0.1307],
}
std_dataset  = {
    "cifar10" : [0.2023, 0.1994, 0.2010],
    "mnist" : [0.3081],
    "fmnist" : [0.3081],
}

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

def reduce_params(sources, weights):
    targets = []
    for i in range(len(sources[0])):
        target = torch.sum(weights * torch.stack([source[i].cuda() for source in sources], dim = -1), dim=-1)
        targets.append(target)
    return targets

def epoch(mode, dataloader, net, optimizer, criterion, aug=True, args=None):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.cuda()
    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().cuda()
        lab = datum[1].cuda()
        if aug:
            img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
        n_b = lab.shape[0]
        output = net(img)
        loss = criterion(output, lab)
        if mode == 'train':
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), np.argmax(lab.cpu().data.numpy(), axis=-1)))
        else:
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset(it_eval, net, lr_net, images_train, labels_train, testloader, args):
    net = net.cuda()
    images_train = images_train.cuda()
    labels_train = labels_train.cuda()
    lr = float(lr_net)
    Epoch = 500
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=0)
    start = time.time()
    acc_train_list = []
    loss_train_list = []
    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, kd_loss, aug=True, args=args)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, nn.CrossEntropyLoss().cuda(), aug=False, args=args)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    time_train = time.time() - start
    print('Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))
    return net, acc_train_list, acc_test

class Synthesizer:
    def __init__(self, network, test_loader, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset =args.dataset
        self.testloader =test_loader
        self.batch_syn = args.batch_syn
        self.save_path = args.RESULTS_PATH
        self.iteration = args.Iteration
        self.channel = args.channel
        hard_label = [np.ones(args.ipc, dtype=np.long)*i for i in range(args.num_classes)]
        label_syn = torch.nn.functional.one_hot(torch.tensor(hard_label).reshape(-1), num_classes=args.num_classes).float()
        label_syn = label_syn * args.label_init
        label_syn = label_syn.detach().to(self.device).requires_grad_(True)
        image_syn = torch.randn(size=(args.num_classes * args.ipc, args.channel, args.imsize[0], args.imsize[1]), dtype=torch.float)
        syn_lr = torch.tensor(args.lr_teacher).to(self.device)
        image_syn = image_syn.detach().to(self.device).requires_grad_(True)
        syn_lr = syn_lr.detach().to(self.device).requires_grad_(True)
        if args.img_optim == "sgd":
            optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
            optimizer_label = torch.optim.SGD([label_syn], lr=args.lr_label, momentum=0.5)
        else:
            optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
            optimizer_label = torch.optim.Adam([label_syn], lr=args.lr_label)
        if args.lr_optim == "sgd":
            optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
        else:
            optimizer_lr = torch.optim.Adam([syn_lr], lr=args.lr_lr)
        self.test_loader = test_loader
        self.label_syn, self.image_syn, self.syn_lr = label_syn, image_syn, syn_lr
        self.optimizer_img, self.optimizer_label, self.optimizer_lr = optimizer_img, optimizer_label, optimizer_lr
        self.network = network.cuda()
        self.weight_averaging, self.least_ave_num, self.max_ave_num, self.random_weights = args.weight_averaging, args.least_ave_num, args.max_ave_num, args.random_weights
        self.distributed = torch.cuda.device_count() > 1
        self.syn_steps, self.min_start_epoch , self.max_start_epoch, self.expert_epochs = args.syn_steps, args.min_start_epoch, args.max_start_epoch, args.expert_epochs


    def synthesize(self, trajectories_list, args):
        for it in range(0, self.iteration):
            trajectories = trajectories_list[random.randint(0, len(trajectories_list)-1)]
            # trajectories = trajectories_list[-1]
            student_net = ReparamModule(copy.deepcopy(self.network))
            if self.distributed:
                student_net = torch.nn.DataParallel(student_net)
            student_net.train()
            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
            curr_max_start_epoch = min([self.max_start_epoch, len(trajectories) - 1 - self.expert_epochs])
            if curr_max_start_epoch == 0:
                start_epoch = 0
            else:
                start_epoch = np.random.randint(self.min_start_epoch, curr_max_start_epoch+1)
            # print(f"max start epoch {curr_max_start_epoch}, min start epoch {self.min_start_epoch}, expert epoch {self.expert_epochs}")
            # print(f"sampled  start epoch {start_epoch}")
            starting_params = trajectories[start_epoch]
            if not self.weight_averaging:
                target_params = trajectories[start_epoch+self.expert_epochs]
            else:
                max_ave_num = self.max_ave_num+1 if self.max_ave_num < self.expert_epochs else self.expert_epochs+1
                averaging_num = random.choice(list(range(self.least_ave_num, max_ave_num)))
                candidate_params = random.choices(trajectories[start_epoch+1: start_epoch+self.expert_epochs+1],k=averaging_num)
                if not self.random_weights:
                    weights = torch.full([len(candidate_params)], 1./len(candidate_params), dtype=torch.float, device="cuda")
                else:
                    weights = torch.rand(len(candidate_params)).to(self.device)
                    weights = torch.softmax(weights, dim=0)
                target_params = reduce_params(candidate_params, weights)

            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)
            student_params = [torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)
            syn_images = self.image_syn
            y_hat = self.label_syn
            param_loss_list = []
            param_dist_list = []
            indices_chunks = []

            for step in range(self.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(syn_images))
                    indices_chunks = list(torch.split(indices, self.batch_syn))

                these_indices = indices_chunks.pop()
                x = syn_images[these_indices]
                this_y = y_hat[these_indices]
                if args.dsa:
                    x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
                if self.distributed:
                    forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params = student_params[-1]
                x = student_net(x, flat_param=forward_params)
                ce_loss = kd_loss(x, this_y)
                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
                student_params.append(student_params[-1] - self.syn_lr * grad)

            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)


            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss

            self.optimizer_img.zero_grad()
            self.optimizer_label.zero_grad()
            self.optimizer_lr.zero_grad()

            grand_loss.backward()

            self.optimizer_img.step()
            self.optimizer_lr.step()
            self.optimizer_label.step()

            # wandb.log({"Grand_Loss": grand_loss.detach().cpu()})

            for _ in student_params:
                del _

            if it%10 == 0:
                print('iter = %04d, loss = %.4f' % (it, grand_loss.item()))
                print(f"syn_labels = {F.softmax(self.label_syn)}")
            if (it+1)%500 == 0:
                self.evaluate(0,upload_wandb=False, args=args)

    def evaluate(self, c_round, upload_wandb= True, args=None):
        accs_test = []
        accs_train = []
        for it_eval in range(3):
            net_eval = copy.deepcopy(self.network).to(self.device) # get a random model
            eval_labs = self.label_syn.detach()
            with torch.no_grad():
                image_save = self.image_syn
            image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification
            lr_net = self.syn_lr.item()
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, lr_net, image_syn_eval, label_syn_eval, self.testloader, args)
            accs_test.append(acc_test)
            accs_train.append(acc_train)
        accs_test = np.array(accs_test)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        print('Evaluate %d, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), acc_test_mean, acc_test_std))

        # uploading images to wandb
        upsampled = image_save
        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
        if upload_wandb:
            wandb.log({'Accuracy/{}'.format("ConvNet"): acc_test_mean}, step=c_round)
            wandb.log({'Std/{}'.format("ConvNet"): acc_test_std}, step=c_round)
            wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=c_round)
            wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=c_round)

        image_syn_vis = copy.deepcopy(image_save.detach().cpu())
        for ch in range(self.channel):
            image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std_dataset.get(self.dataset)[ch] + mean_dataset.get(self.dataset)[ch]
        image_syn_vis[image_syn_vis<0] = 0.0
        image_syn_vis[image_syn_vis>1] = 1.0
        grid = torchvision.utils.make_grid(image_syn_vis, nrow=10, normalize=True, scale_each=True)
        wandb.log({"Synthetic vis_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=c_round)
        save_data_path = os.path.join(self.save_path, "syn_data")
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        torch.save(self.image_syn.detach().cpu(), os.path.join(save_data_path, "images_best.pt".format(c_round)))
        torch.save(self.label_syn.detach().cpu(), os.path.join(save_data_path, "labels_best.pt".format(c_round)))
        print(f"saved synthetic data at {save_data_path}")

