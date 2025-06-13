import torch
from torchvision import models, transforms
from PIL import Image
import urllib

def load_places365():
    model = models.resnet50(num_classes=365)
    checkpoint = torch.hub.load_state_dict_from_url(
        'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar', map_location='cpu'
    )
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def classify_scene(img_path):
    model = load_places365()
    with open('categories_places365.txt') as f:
        classes = [line.strip().split(' ')[0][3:] for line in f.readlines()]

    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_img = transform(img).unsqueeze(0)
    output = model(input_img)
    scene_index = torch.argmax(output)
    scene_name = classes[scene_index]
    return scene_name
