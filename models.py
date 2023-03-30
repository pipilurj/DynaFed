import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, conv3x3, conv1x1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

channel_dict =  {
    "cifar10": 3,
    "cinic10": 3,
    "cifar100": 3,
    "mnist": 1,
    "fmnist": 1,
}

############################################################################################################
# MOBILENET
############################################################################################################

class MLP(nn.Module):
    def __init__(self, num_classes=10, net_width=128, im_size = (28,28), dataset = 'cifar10'):
        super(MLP, self).__init__()
        channel = channel_dict.get(dataset)
        self.fc1 = nn.Linear(im_size[0]*im_size[1]*channel, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.fc3 = nn.Linear(net_width, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, norm_layer):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)

    def __init__(self, num_classes=10, norm_layer=nn.BatchNorm2d,shrink=1, dataset = 'cifar10'):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.dataset = dataset
        channel =  channel_dict.get(dataset)
        self.norm_layer = norm_layer
        self.cfg = [(1,  16//shrink, 1, 1),
                   (6,  24//shrink, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                   (6,  32//shrink, 3, 2),
                   (6,  64//shrink, 4, 2),
                   (6,  96//shrink, 3, 1),
                   (6, 160//shrink, 3, 2),
                   (6, 320//shrink, 1, 1)]


        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(self.cfg[-1][1], 1280//shrink, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = self.norm_layer(1280//shrink)


        self.classification_layer = nn.Linear(1280//shrink, num_classes)


    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, self.norm_layer))
                in_planes = out_planes
        return nn.Sequential(*layers)


    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


    def forward(self, x):
        feature = self.extract_features(x)
        out = self.classification_layer(feature)
        return out




def mobilenetv2(num_classes=10, dataset = 'cifar10'):
    return MobileNetV2(norm_layer=nn.BatchNorm2d, shrink=2, num_classes=num_classes, dataset = 'cifar10')


    

############################################################################################################
# RESNET
############################################################################################################
class basic_noskip(BasicBlock):
    expansion: int = 1
    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        super(basic_noskip, self).__init__(*args, **kwargs)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        return out

class Model_noskip(nn.Module):
    def __init__(self, channel=3, feature_dim=128, group_norm=False):
        super(Model_noskip, self).__init__()

        self.f = []
        for name, module in ResNet(basic_noskip, [1,1,1,1], num_classes=10).named_children():
            if name == 'conv1':
                module = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        if group_norm:
            apply_gn(self)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class resnet8_noskip(nn.Module):
    def __init__(self, num_classes=10, pretrained_path=None, group_norm=False, dataset = 'cifar10'):
        super(resnet8_noskip, self).__init__()
        channel =  channel_dict.get(dataset)
        # encoder
        self.f = Model_noskip(channel = channel, group_norm=group_norm).f
        # classifier
        self.classification_layer = nn.Linear(512, num_classes, bias=True)


        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)


    def extract_features(self, x):
        return torch.flatten(self.f(x), start_dim=1)


    def forward(self, x):
        feature = self.extract_features(x)
        out = self.classification_layer(feature)
        return out

class Model(nn.Module):
    def __init__(self, channel=3, feature_dim=128, group_norm=False):
        super(Model, self).__init__()

        self.f = []
        for name, module in ResNet(BasicBlock, [1,1,1,1], num_classes=10).named_children():
            if name == 'conv1':
                module = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        if group_norm:
            apply_gn(self)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class resnet8(nn.Module):
    def __init__(self, num_classes=10, pretrained_path=None, group_norm=False, dataset = 'cifar10'):
        super(resnet8, self).__init__()
        channel =  channel_dict.get(dataset)
        # encoder
        self.f = Model(channel = channel, group_norm=group_norm).f
        # classifier
        self.classification_layer = nn.Linear(512, num_classes, bias=True)


        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)


    def extract_features(self, x):
        return torch.flatten(self.f(x), start_dim=1)


    def forward(self, x):
        feature = self.extract_features(x)
        out = self.classification_layer(feature)
        return out




############################################################################################################
# SHUFFLENET
############################################################################################################



class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes//4
        g = 1 if in_planes==24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ShuffleNet, self).__init__()
        cfg = {'out_planes': [200,400,800],'num_blocks': [4,8,4],'groups': 2}

        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.classification_layer = nn.Linear(out_planes[2], num_classes)
     

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)


    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        return feature


    def forward(self, x):
        feature = self.extract_features(x)
        out = self.classification_layer(feature)
        return out


''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size = (32,32), dataset = 'cifar10'):
        super(ConvNet, self).__init__()
        channel =  channel_dict.get(dataset)
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        print(f"num feat {num_feat}")
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.get_feature(x)
        out = self.classifier(out)
        return out

    def get_feature(self,x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat



class TextModel(nn.Module):

    def __init__(self, vocab_size=95811, embed_dim=64, num_classes=4):
        super(TextModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=130107, num_classes=20):
        super(LogisticRegression, self).__init__()
        self.fc = torch.nn.Parameter(torch.zeros(input_dim, num_classes))


    def forward(self, x):
        out = x @ self.fc
        return out

def get_model(model):

  return  {   "mobilenetv2" : (mobilenetv2, optim.Adam, {"lr" : 0.001}),
              "shufflenet" : (ShuffleNet, optim.Adam, {"lr" : 0.001}),
                "resnet8" : (resnet8, optim.Adam, {"lr" : 0.001}),
                "resnet8_noskip" : (resnet8_noskip, optim.Adam, {"lr" : 0.001}),
                "ConvNet" : (ConvNet, optim.Adam, {"lr" : 0.001}),
                "MLP" : (MLP, optim.Adam, {"lr" : 0.001}),
                "TextModel" : (TextModel, optim.Adam, {"lr" : 1}),
                "LogisticRegression" : (LogisticRegression, optim.Adam, {"lr" : 0.001}),
          }[model]


def print_model(model):
  n = 0
  print("Model:")
  for key, value in model.named_parameters():
    print(' -', '{:30}'.format(key), list(value.shape), "Requires Grad:", value.requires_grad)
    n += value.numel()
  print("Total number of Parameters: ", n) 
  print()






