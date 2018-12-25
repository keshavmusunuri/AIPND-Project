import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from network import Network
from utils import save_model


def create_arguments():
    parser = argparse.ArgumentParser(description="create Model")
    archs = ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'vgg13', 'vgg16', 'vgg19']

    parser.add_argument('--data_dir', action='store', default='flowers', help='Directory to get source images')
    parser.add_argument('--gpu', action="store_true", default=True)
    parser.add_argument('--arch', dest='arch', default='densenet201', choices=archs,
                        help='choose the desired model architecture')
    parser.add_argument('--hidden_units', dest='hidden_units', default= 512, nargs='+',
                        help='choose the desired hidden layer architecture')
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.01',
                        help='choose the desired learning rate')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int, help='choose the desired number of epochs')
    parser.add_argument('--save_dir', action='store', help='Directory to save checkpoints')

    return parser.parse_args()


def validate(model, criterion, loader, use_cuda):
    test_loss = 0
    accuracy = 0
    if use_cuda:
        model.cuda()
    for images, labels in loader:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equality = (top_class == labels.view(*top_class.shape))
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    return test_loss, (accuracy * 100)


def train(model, criterion, optimizer, train_loader, valid_loader, epochs, use_cuda):
    print_every = 25
    steps = 0

    if use_cuda:
        model.cuda()

    for e in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in train_loader:
            steps += 1
            optimizer.zero_grad()
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    validation_loss, accuracy = validate(model, criterion, valid_loader, use_cuda)

                print("\nEpoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss / len(valid_loader)),
                      "Validation Accuracy: {:.2f} %".format(accuracy / len(valid_loader)))

                running_loss = 0
                model.train()


def construct_model():
    args = create_arguments()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    try:
        input_size = model.classifier.in_features
    except:
        input_size = 25088
    hidden_sizes = list(map(int, args.hidden_units))

    output_size = 102
    dropout = 0.5

    classifier = Network(input_size, hidden_sizes, output_size, dropout)
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))

    gpu = args.gpu
    cuda = torch.cuda.is_available()
    use_cuda = gpu and cuda

    epochs = args.epochs

    print(model)
    train(model, criterion, optimizer, train_loader, valid_loader, epochs, use_cuda)
    model.class_to_idx = train_dataset.class_to_idx
    save_model(args, classifier)


if __name__ == "__main__":
    construct_model()