import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
import pickle
import torchvision.transforms as transforms
import sys

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
image_transforms = transforms.Compose([
                           transforms.Resize(IMAGE_SIZE),
                           transforms.CenterCrop(IMAGE_SIZE),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = MEAN, std = STD)])

index2name = pickle.load(open("imagenet_class_names.pkl", "rb"))

cuda = torch.cuda.is_available()
model = torchvision.models.resnet50(pretrained=True)
softmax_layer = nn.Softmax(dim=1)
model.eval()
if cuda:
  model.cuda()
print("Loaded Model Successfully")

image = Image.open(sys.argv[1])
image = image_transforms(image)
image = image.unsqueeze(0)

if cuda:
  image = image.cuda()
output = softmax_layer(model(image))

prediction = torch.argmax(output, dim=1).item()
prob = output[0, prediction].item()
predicted_name = index2name[prediction]

print("Class: %s. Probabilty: %.2f." % (predicted_name, prob))

