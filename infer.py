import os
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F

from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoConfig, get_cosine_schedule_with_warmup

from models.dinov3_linear import DinoV3Linear
from models.resnet_linear import ResNetLinear


# the architecture to use
num_classes = 365
arch = 'resnet18'
img_name = r"D:\CS\MyProjects\resources\Datasets\CUB_200_2011\val\026.Bronzed_Cowbird\Bronzed_Cowbird_0022_796221.jpg"
resume = 'checkpoints/resnet18_Epoch20_69.292_pure.pth'
categories_file_name = './docs/categories_places365.txt'

# create the model
print(f"[Creating model '{arch}']")
model = None
if arch == 'dinov3':
    MODEL_NAME_OR_PATH = "./checkpoints/weights/dinov3-vits16-pretrain-lvd1689m"
    backbone = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
    model = DinoV3Linear(backbone, num_classes)
elif arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
    backbone = models.__dict__[arch](num_classes=365)
    model = ResNetLinear(backbone, num_classes) # freze backbone
else:
    raise ValueError(f"Unsupported architecture '{arch}'")


# load the model weights
if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint.items()} # checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
else:
    raise FileNotFoundError(f"No checkpoint found at '{resume}'")


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize(int(224 * 1.2)),
        trn.CenterCrop(224),
        # trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label

if not os.access(categories_file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url + ' -P ./docs/')
classes = list()
with open(categories_file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the test image
img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0))

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

print('{} prediction on {}'.format(arch, img_name))
# output the prediction
for i in range(0, 5):
    print(f'{probs[i]:.3f} -> {classes[idx[i]]}, idx: {idx[i]+1}')
