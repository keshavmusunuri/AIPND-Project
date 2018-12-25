import torch
from torchvision import transforms, datasets
import argparse
from PIL import Image
import json

def save_model(model, args, classifier):
    
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier' : classifier,
                  'epochs': args.epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
    
def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]       
    pil_image = Image.open(image).convert("RGB")
    
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])
    return image_transforms(pil_image)

def get_category_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names