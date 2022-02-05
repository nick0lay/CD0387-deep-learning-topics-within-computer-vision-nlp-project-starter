#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from PIL import ImageFile

import argparse

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects / len(test_loader)
    
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, total_acc, len(test_loader.dataset), 100.0 * running_corrects / len(test_loader.dataset)
        )
    )

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=1
    best_loss=1e6
    loss_counter=0
    
    image_dataset={'train': train_loader, 'valid': validation_loader}
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            running_samples=0
            
            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                if running_samples % 5 == 0:
                    print("-------")
                    print(preds)
                    print(labels.data)
                    print(running_corrects)
                running_samples += len(inputs)
                
                if running_samples % 100  == 0 or running_samples == len(image_dataset[phase].dataset):
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
            
            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
        if loss_counter==1:
            break
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Linear(num_features, 133)
    
    # check if CUDA is available
    use_cuda = torch.cuda.is_available()
    
    # move model to GPU if CUDA is available
    if use_cuda:
        model = model.cuda()
        print("GPU available for training")

    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_data = datasets.ImageFolder(data, transform=data_transform)
    classes = image_data.classes
    classes_count = len(classes)
    print(classes_count)
    return  torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True)

def main(args):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader = create_data_loaders(args.train, args.batch_size)
    valid_loader = create_data_loaders(args.validation, args.batch_size)
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = create_data_loaders(args.validation, args.batch_size)
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.to(device).state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
        help="training data input (default: set in env variables)"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=os.environ['SM_CHANNEL_TEST'],
        help="test data input (default: set in env variables)"
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=os.environ['SM_CHANNEL_VALIDATION'],
        help="validation data input (default: set in env variables)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="directory to save model data (default: set in env variables)"
    )
    args=parser.parse_args()
    
    main(args)
