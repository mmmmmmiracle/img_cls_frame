import torch
import torch.nn as nn
import torchvision.models as models 
from efficientnet_pytorch import EfficientNet


def create_model(model_name = 'resnet18', num_classes = 206, pretrained = True):
    # create model
    model = None
    print("=> creating model '{}'".format(model_name))
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        dim_feats = model.fc.in_features
        model.fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        dim_feats = model.fc.in_features
        model.fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        dim_feats = model.fc.in_features
        model.fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'efficientnet_b0':
        model = EfficientNet.from_pretrained(model_name='efficientnet-b0')
        # print(model)
        dim_feats = model._fc.in_features
        model._fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'efficientnet_b3':
        model = EfficientNet.from_pretrained(model_name='efficientnet-b3')
        dim_feats = model._fc.in_features
        model._fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'efficientnet_b5':
        model = EfficientNet.from_pretrained(model_name='efficientnet-b5')
        dim_feats = model._fc.in_features
        model._fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
        dim_feats = model.fc.in_features
        model.fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'resnext101_32x8d':
        model = models.resnext101_32x8d(pretrained=True)
        dim_feats = model.fc.in_features
        model.fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2(pretrained=True)
        dim_feats = model.fc.in_features
        model.fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'wide_resnet101_2':
        model = models.wide_resnet101_2(pretrained=True)
        dim_feats = model.fc.in_features
        model.fc = nn.Linear(dim_feats, num_classes)
    if model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        # print(model)
        dim_feats = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(dim_feats, num_classes)
    if model_name == 'densenet169':
        model = models.densenet169(pretrained=True)
        # print(model)
        dim_feats = model.classifier.in_features
        model.classifier = nn.Linear(dim_feats, num_classes)
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        print(model)
        dim_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(dim_feats, num_classes)
    else:
        print("Please add more pretrained models, we have not much enough yet")
    return model