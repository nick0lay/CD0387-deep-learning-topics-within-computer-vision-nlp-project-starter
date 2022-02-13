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
from PIL import ImageFile
from PIL import Image

import argparse
import json
import logging
import os
import sys
import io
# import smdebug.pytorch as smd
# from smdebug.pytorch import get_hook

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


#TODO: Import dependencies for Debugging andd Profiling

def test(model, loader, criterion, device):
#     hook = get_hook(create_if_not_exists=True)
    
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
#     hook.set_mode(smd.modes.PREDICT)

    running_loss=0
    running_corrects=0
    running_samples=0
    
    for inputs, labels in loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        running_samples += len(inputs)

    total_loss = running_loss / running_samples
    total_acc = running_corrects / running_samples
    
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, running_samples, 100.0 * total_acc
        )
    )

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
#     hook = get_hook(create_if_not_exists=True)
    
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss=1e6
    loss_counter=0
    
    image_dataset={'train': train_loader, 'valid': validation_loader}
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
#                 hook.set_mode(smd.modes.TRAIN)
                model.train()
            else:
#                 hook.set_mode(smd.modes.EVAL)
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
    
def net(model):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    print(f"Create model {model}")
    model = models.__dict__[model](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
#     model.fc = nn.Linear(num_features, 133)
    num_classes = 133
    model.fc = nn.Sequential(nn.Linear(num_features,512),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(512, num_classes))
    
    # check if CUDA is available
    use_cuda = torch.cuda.is_available()
    
    # move model to GPU if CUDA is available
    if use_cuda:
        model = model.cuda()
        print("GPU available for training")

    return model

def create_data_loaders(data, batch_size, transformer, is_shuffle = False):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    image_data = datasets.ImageFolder(data, transform=transformer)
    print(len(image_data.classes))
    return  torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=is_shuffle)

def model_fn(model_dir):
    logger.info(f"Creating model from {model_dir}")
    model = net("resnet34")
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


def save_model(model, model_dir):
    logger.info(f"Saving the model in {model_dir}.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.to(device).state_dict(), path)
    
def predict_fn(input_data, model):
    logger.info("Predict dog breed")
    logger.info(type(input_data))
    logger.info(input_data)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_data = torch.unsqueeze(input_data, 0)
        return model(input_data.to(device))
    
def input_fn(request_body, request_content_type):
    logger.info("Model input")
    logger.info(type(request_body))
    logger.info(request_content_type)
    logger.info(request_body)
    
    data = np.load(io.BytesIO(request_body))
    data = data.tolist()
    logger.info("Model data")
    logger.info(type(data))
    logger.info(data)
    
    img_dimension = 224
    transform = transforms.Compose([
        transforms.Resize(img_dimension),
        transforms.CenterCrop(img_dimension),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_data = Image.open(io.BytesIO(data))
    input_data = transform(input_data)
    return input_data
    
def output_fn(prediction, content_type):
    logger.info("Dog breed predicted")
    logger.info(type(prediction))
    logger.info(prediction)
    logger.info(content_type)
    
    values, indexes = torch.max(prediction, 1)
    value = indexes
    print(f"Predicted value {value}")
#     return value # works
    value = value.cpu().numpy().astype(str)
    return json.dumps(list(value)), "application/json"

# def transform_fn(model, request_body, request_content_type, response_content_type):
#     """Run prediction and return the output.
#     The function
#     1. Pre-processes the input request
#     2. Runs prediction
#     3. Post-processes the prediction output.
#     """
#     logger.info("Transform function")
#     logger.info(response_content_type)
#     logger.info(request_body)
#     # preprocess
#     decoded = Image.open(io.BytesIO(request_body))
#     preprocess = transforms.Compose([
#         transforms.Resize(img_dimension),
#         transforms.CenterCrop(img_dimension),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     normalized = preprocess(decoded)
#     batchified = normalized.unsqueeze(0)
    
#     # predict
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     batchified = batchified.to(device)
#     output = model(normalized)
#     values, indexes = torch.max(output, 1)
#     return int(indexes[0])

def main(args):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
#     hook = get_hook(create_if_not_exists=True)
#     hook = smd.Hook.create_from_json_file()
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net(args.model)
#     hook.register_hook(model)
    save_model(model, args.model_dir)
    
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
    img_dimension = 224
    
    train_transformer = transforms.Compose([
        transforms.Resize(img_dimension),
        transforms.CenterCrop(img_dimension),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_loader = create_data_loaders(args.train, args.batch_size, train_transformer, True)
    
    valid_transformer = transforms.Compose([
        transforms.Resize(img_dimension),
        transforms.CenterCrop(img_dimension),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_loader = create_data_loaders(args.validation, args.batch_size, valid_transformer)
    
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = create_data_loaders(args.test, args.batch_size, valid_transformer)
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.to(device).state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
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
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 3)",
    )
    args=parser.parse_args()
    
    main(args)
