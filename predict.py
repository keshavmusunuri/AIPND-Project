import torch
from torchvision import transforms, models
import argparse
import numpy as np
import json
from utils import load_model, process_image, get_category_names

def create_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', dest='filepath', default=None, help='File Path to get image')
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth',help='checkpoint file name' )
    parser.add_argument('--top_k', dest='top_k', default='5', help='total number of top classes required')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='category names json file')
    parser.add_argument('--gpu', action='store_true', default=True, help='Run on GPU')
    
    return parser.parse_args()


def predict(image_path, model, topk, gpu):

    model.cpu()
    model.eval()
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        
        top_p, top_class = ps.topk(topk)
        probable_classes = []
        
        for label in top_class.numpy()[0]:
            probable_classes.append(list(model.class_to_idx.keys())[list(model.class_to_idx.values()).index(label)])
    return top_p.numpy()[0], probable_classes

def main(): 
    args = create_arguments()
    gpu = args.gpu
    cuda = torch.cuda.is_available()
    use_cuda = gpu and cuda
    model = load_model(args.checkpoint)
    category_names = get_category_names(args.category_names)

    img_path = args.filepath
    top_p, probable_classes = predict(img_path, model, args.top_k, gpu)
 
    print(top_p)
    print(probable_classes)
    print([category_names[category_name] for category_name in probable_classes])

if __name__ == "__main__":
    main()