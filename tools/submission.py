import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as transforms
from torch.nn import functional as F
import torch.nn as nn
import os
from PIL import Image
import time
import json
import argparse
import cv2
from mmcv import Config
import sys
sys.path.append('.')
from utils.utils import SceneData

parser = argparse.ArgumentParser(
    description="Scene image testing")


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('config', help='train config file path')
parser.add_argument('checkpoint', type=str, help='trained model')
args = parser.parse_args()
cfg = Config.fromfile(args.config)

# arch = cfg.model.arch
def main():
    # create model
    print("=> loading model '{}'".format(cfg.model.arch))
    if cfg.model.arch == 'resnet18':
        model = models.resnet18(pretrained=False)
        dim_feats = model.fc.in_features
        model.fc = nn.Linear(dim_feats, cfg.model.num_classes)
        checkpoint = torch.load(args.checkpoint)
    else:
        print("Please add more pretrained models, we have not much enough yet")
        return 0

    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

    model.load_state_dict(state_dict)

    model.cuda()
    model.eval()

    # prepare test data
    print("=> loading test data...")

    valdir = os.path.join(cfg.data, 'val')

    with open(cfg.annotations.test, 'r') as f:
        datasets = f.readlines()

    # inference
    print("=> inference...")

    val_tf = transforms.Compose([
             transforms.Resize(size=256),
             transforms.CenterCrop(size=224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
             ])
    result_json = []
    with torch.no_grad():
        for data in datasets:
            data = data.rstrip()
            path = os.path.join(valdir, data)
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            result = {}
            image = Image.open(path)
            image = V(val_tf(image).unsqueeze(0))
            image = image.cuda()
            logit = model.forward(image)
            h_x = F.softmax(logit, 1).data.squeeze()
            _, idx = h_x.sort(0, True)
            pred_label = idx[0].item()
            result["type"] = 0
            result["image_id"] = data
            result["category_id"] = pred_label
            result_json.append(result)

    with open("results/submission.json", "w") as f:
        f.write(json.dumps(result_json))

if __name__ == '__main__':
    main()
