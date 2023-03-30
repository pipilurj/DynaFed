import random
import models as model_utils
from utils import *
from client import Device
from utils import kd_loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Server(Device):
  def __init__(self, model_names, loader, val_loader, num_classes=10, images_train=None, labels_train=None, eta=0.5 , dataset = 'cifar10', client_loaders=None):
    super().__init__(loader)
    self.val_loader = val_loader

    print(f"dataset server {dataset}")
    self.model_dict = {model_name : partial(model_utils.get_model(model_name)[0], num_classes=num_classes, dataset = dataset)().to(device) for model_name in model_names}
    self.parameter_dict = {model_name : {key : value for key, value in model.named_parameters()} for model_name, model in self.model_dict.items()}
    self.client_loaders = client_loaders
    self.images_train, self.labels_train = images_train, labels_train
    self.eta = eta

    
    self.models = list(self.model_dict.values())


  def evaluate_ensemble(self):
    return eval_op_ensemble(self.models, self.loader, self.val_loader)


  def select_clients(self, clients, frac=1.0, unbalance_rate=1, sample_mode="uniform"):
    return random.sample(clients, int(len(clients)*frac))

  def fedavg(self, clients):
    unique_client_model_names = np.unique([client.model_name for client in clients])
    self.weights = torch.Tensor([1. / len(clients)] * len(clients))
    for model_name in unique_client_model_names:
      reduce_average(target=self.parameter_dict[model_name], sources=[client.W for client in clients if client.model_name == model_name])

  def distill(self, clients, optimizer_fn, epochs=1, mode="mean_logits", num_classes=10):
    optimizer_dict = {model_name: optimizer_fn(
        model.parameters()) for model_name, model in self.model_dict.items()}
    for model_name in self.model_dict:
      print("Distilling {} ...".format(model_name))

      model = self.model_dict[model_name]
      optimizer = optimizer_dict[model_name]

      model.train()

      for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x,_ in tqdm(self.val_loader):
          x = x.to(device)

          if mode == "mean_logits":
            y = torch.zeros([x.shape[0], num_classes], device="cuda")
            for i, client in enumerate(clients):
              y_p = client.predict_logit(x)
              y += (y_p/len(clients)).detach()

            y = nn.Softmax(1)(y)

          optimizer.zero_grad()

          y_ = nn.LogSoftmax(1)(model(x))

          loss = torch.nn.KLDivLoss(reduction="batchmean")(y_, y.detach())

          running_loss += loss.item()*y.shape[0]
          samples += y.shape[0]

          loss.backward()
          optimizer.step()

    return {"loss": running_loss / samples, "epochs": ep}


  def abavg(self, clients):
    unique_client_model_names = np.unique([client.model_name for client in clients])
    acc = torch.zeros([len(clients)], device="cuda")
    for x, true_y in self.val_loader:
      x = x.to(device)
      true_y = true_y.to(device)
      samples = x.shape[0]
      for i, client in enumerate(clients):
        y_ = client.predict_logit(x)
        _, predicted = torch.max(y_.detach(), 1)
        acc[i] = (predicted == true_y).sum().item()/ samples
    self.weights = acc/ acc.sum()
    print(self.weights)
    for model_name in unique_client_model_names:
      reduce_weighted(target=self.parameter_dict[model_name], sources=[client.W for client in clients if client.model_name == model_name], weights = self.weights)


  def feddf(self, clients, distill_iter, distill_optimizer_fn, num_classes):
    unique_client_model_names = np.unique(
        [client.model_name for client in clients])
    self.weights = torch.Tensor([1. / len(clients)] * len(clients))
    for model_name in unique_client_model_names:
      reduce_average(target=self.parameter_dict[model_name], sources=[
                     client.W for client in clients if client.model_name == model_name])
    self.distill(clients, distill_optimizer_fn,
                 distill_iter, "mean_logits", num_classes)

  def datadistill(self, clients, distill_iter, distill_lr, dsa, args, current_round=0, start_round=0, ifsoft=True, test_client = False):
    if self.images_train is None or self.labels_train is None or current_round < start_round:
      self.fedavg(clients)
    else:
      unique_client_model_names = np.unique(
        [client.model_name for client in clients])
      for model_name in unique_client_model_names:
        reduce_average(target=self.parameter_dict[model_name], sources=[
          client.W for client in clients if client.model_name == model_name])
      distilled_dataset = TensorDataset(self.images_train, self.labels_train)
      distilled_loader = torch.utils.data.DataLoader(distilled_dataset, batch_size=256, shuffle=True)
      client_test_losses = [[], [], []]
      print(f"num of loaders {len(clients)}")
      for model_name in self.model_dict:
        model = self.model_dict[model_name]
        model.train()
        with torch.no_grad():
          if args.pass_forward:
            for _ in range(3):
              for (x_dis, y_dis) in distilled_loader:
                x_dis , y_dis = x_dis.to(device), y_dis.to(device)
                model(x_dis)
        optimizer = torch.optim.Adam(model.parameters(), lr=distill_lr)
        loss_avg = 0
        for _ in range(distill_iter):
          if test_client:
            with torch.no_grad():
              model.eval()
              for i, client_loader in enumerate(self.client_loaders):
                samples, correct, loss_c = 0, 0, 0
                for x_c, y_c in client_loader:
                  x_c, y_c  = x_c.to(device), y_c.to(device)
                  out_c = model(x_c)
                  _, predicted = torch.max(out_c.detach(), 1)
                  l = F.cross_entropy(out_c, y_c).item()*y_c.shape[0]
                  samples += y_c.shape[0]
                  loss_c += l
                test_loss_c = loss_c/samples
                client_test_losses[i].append(round(test_loss_c, 2))
          model.train()
          for (x_dis, y_dis) in distilled_loader:
            x_dis , y_dis = x_dis.to(device), y_dis.to(device)
            if dsa:
              x_dis = DiffAugment(x_dis, args.dsa_strategy, param=args.dsa_param)
            optimizer.zero_grad()
            if ifsoft:
              loss_distill = kd_loss(model(x_dis), y_dis)
            else:
              loss_distill = nn.CrossEntropyLoss()(model(x_dis), y_dis)
            loss_distill.backward()
            loss_avg += loss_distill.item()
            optimizer.step()
        print("Server client losses:")
        print(client_test_losses)
        print(f"length of client losses {[len(x) for x in client_test_losses]}")


  def sync_bn(self):
    for model in self.models:
      model.train()
      for x, _ in self.val_loader:
        x = x.to(device)
        y = model(x)


