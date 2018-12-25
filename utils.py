import torch
from torchvision import transforms, datasets, models
import argparse
from PIL import Image
import json

def save_model(args, classifier):
    
    checkpoint = {'arch': args.arch, 
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier' : classifier,
                  'epochs': args.epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
    
def load_model(filepath):
    checkpoint = torch.load(filepath)
    model =  getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):  
    image = Image.open(image_path)
    size = 255
    width, height = image.size
    if height > width:
        height = int(max(height * 255 / width, 1))
        width = int(255)
    else:
        width = int(max(width * 255 / height, 1))
        height = int(255)            
    resized_image = image.resize((width, height))
    
    size = 224
    width, height = resized_image.size
    x0 = (width - 224) / 2
    y0 = (height - 224) / 2
    x1 = x0 + 224
    y1 = y0 + 224
    cropped_image = resized_image.crop((x0, y0, x1, y1))

    image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    image = (image - mean) / std
        
    image = image.transpose((2, 0, 1))
    
    return image

def get_category_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names