import numpy as np
import torch
from Frame import CNN
from torchvision import transforms
from PIL import Image

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_path = 'food-11/testing/0012.jpg'
model_path = './checkpoint.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img = Image.open(test_path)
img = test_transform(img)
img = img.unsqueeze(0)
img = img.to(device)

net = CNN()
net.to(device)
net.load_state_dict(torch.load(model_path))

label = net(img)
print(np.argmax(label.cpu().data.numpy(), axis=1))
