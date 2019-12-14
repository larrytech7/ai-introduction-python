# PROGRAMMER: Akah harvey
# DATE CREATED: 14/12/2019
# REVISED DATE: 14/12/2019            <=(Date Revised - if any)
# PURPOSE: Nanodegree program for training a CNN model to predict flower images

from myutils import get_input_args
from myutils import get_model_nodes

import time
import sys
import numpy as np
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim

def load_training_data(train_dir,means,deviations,chunk_size):
    #do TRAIN transforms, datasets and loaders
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means,deviations)
    ])
    
    trainingData = datasets.ImageFolder(train_dir, transform=train_transforms)  
    trainingDataloader = torch.utils.data.DataLoader(trainingData, batch_size=chunk_size, shuffle=True)
    
    return trainingDataloader, trainingData

def load_test_data(train_dir,n_means,deviations,chunk_size):
    #do VALIDATE/TEST transforms, datasets and loaders
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(n_means,deviations)
    ])
    
    test_data = datasets.ImageFolder(train_dir, transform=test_transforms)
    
    test_data_load = torch.utils.data.DataLoader(test_data, batch_size=chunk_size, shuffle=True)
    
    return test_data_load

def load_model():
    #parse architecture required from CLI args for the model
    if in_args.arch == "vgg": #default model
        arch_model = models.vgg16(pretrained=True)
        model_nodes = 25088
        model_name = 'vgg16'
    else:
        model_name = in_args.arch.casefold()
        arch_model = eval ("model." + model_name + "(pretrained = True)")
        model_nodes = get_model_nodes(arch_model, model_name)
        
    arch_model.name = model_name
        
    #freeze the params for base Sequential   
    for param in arch_model.parameters():
        param.requires_grad = False
    
    #get out_class nodes number - this is a REQUIRED CLI arg
    class_nodes_count = in_args.output_classes
    
    #get hidden units
    h_units = in_args.hidden_units

    # redeine the model classifier with correct out/in nodes
    arch_model.classifier = nn.Sequential(nn.Linear(model_nodes, h_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(h_units, class_nodes_count),
        nn.LogSoftmax(dim=1)
    )
    
    return arch_model

def train_model(model, device, trainingloader, validDataloaders, epochs, steps, running_loss, display_freq, myOptimizer, criterion):
    trainSuccess = False
    try:
        for rnd in range(epochs):
            for img, labels in trainingloader:
                img, labels = img.to(device), labels.to(device)
                #level out gradients
                myOptimizer.zero_grad()
                loss = criterion(model(img), labels)
                loss.backward()
                myOptimizer.step()
                running_loss += loss.item()

                steps += 1

                #upon 5 steps, do validate checks and display progress
                if steps % display_freq == 0:
                    validLoss = 0
                    precision = 0
                    model.eval()

                    with torch.no_grad():
                        for images, labels in validDataloaders:
                            img, label = images.to(device), labels.to(device)
                            batchLoss = criterion(model(img), label)
                            validLoss += batchLoss.item()

                            #compute accuracy
                            ps = torch.exp(model(img))
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == label.view(*top_class.shape)
                            precision += torch.mean(equals.type(torch.FloatTensor)).item()

                        #show progress
                        print("Round {}/{}".format((rnd+1), epCount), " => ", 
                              "Training loss : {:.2f}". format(running_loss/showFrequency), 
                              "=> Valid Loss : {:.2f}".format(validLoss/len(validDataloaders)), 
                              "=> Valid Accuracy : {:.2f}%".format(100 *(precision/len(validDataloaders))))

                    running_loss = 0
                    #reset model to initial training
                    model.train()
                    trainSuccess = True
    except:
        print ("An error occurred in training function. Check and try again", sys.exc_info()[0])
        print(sys.exc_info())
        trainSuccess = False
        
    return trainSuccess

def test_model(model, device, testloader, test_loss, precision, optimizer, criterion):
    #want no optimizing of weights
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)    
            # Calculate precision
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            precision += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test Precision: {precision/len(testloader)* 100:.3f} %")
    model.train()
    return True

in_args = get_input_args('train') # get all CLI arguments for this program

def main():
    print(in_args) #dictionary of CLI arguments
    data_dir =  in_args.datadir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Build data loaders for the training
    means = lambda x: "0.485, 0.456, 0.406" if x == None else x
    data_means = means(in_args.data_means)
    n_means = list(np.float_(data_means.split(',')))
    print("Normalizing means: {}".format(n_means))

    stdFxn = lambda x: "0.229, 0.224, 0.225" if x == None else x
    data_stds = stdFxn(in_args.data_stds) 
    deviations = list(np.float_(data_stds.split(',')))
    print("Normalizing Standard deviasations: {}".format(deviations))

    chunks = lambda x: 72 if x == None else x
    chunk_size = chunks(in_args.chunks)
    print("Batch size: ", chunk_size)

    #Call the data loader create process
    trainLoader, training_data = load_training_data(train_dir, n_means, deviations, chunk_size) #load images from training directory
    validLoader = load_test_data(valid_dir, n_means, deviations, chunk_size) #load from valid directory
    testingLoader = load_test_data(test_dir, n_means, deviations, chunk_size) #load data from test directory

    #load CNN model#
    model = load_model()
    print("*"*5, "Using Model Architetcure : {}".format(model.name), "*"*5)

    #setup loss function configure learn rate and initialize optimizer
    rate_fxn = lambda x: 0.003 if x == None else x
    learn_rate = rate_fxn(in_args.learning_rate)
    print("Learning rate: {:.3f}".format(learn_rate) )

    criterion = nn.NLLLoss()
    myOptimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)


    # Init device to use. Default : cuda GPU 
    if in_args.device == None: 
        if torch.cuda.is_available(): # if there's no device in the CLI args, we want to use cuda, we need to check if it's present
            torch.cuda.empty_cache()
            training_device = torch.device('cuda')
        else:
            training_device = torch.device('cpu')
    else:
        if in_args.device == 'cuda':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                training_device = torch.device('cuda')
            else:
                training_device = torch.device('cpu')
        else:
            training_device = torch.device(in_args.device)

    model.to(training_device)
    print("Device : ", training_device)

    # Determine epochs and log frequencies during the training.
    epochFxn = lambda x: 5 if x == None else x
    epochs = epochFxn(in_args.epochs)
    print("Epochs: ", epochs)
    startTime = time.time()
    steps = 0
    running_loss = 0
    display_freq = 5

    print ("\n","*"*5,"Training model starting", "*"*5)
    result = train_model(model, training_device, trainLoader, validLoader, epochs, steps, running_loss,display_freq, myOptimizer, criterion)
    print ("\nTraining succeeded ?", result)
    print (f"Device = {training_device}; Total Training Time: {(time.time() - startTime)/60:.3f} minutes")

    #Clear gpu memory let's start new training
    torch.cuda.empty_cache()

    # Execute TEST Run - to verify model on untrained data
    # use built model, device and testloader
    test_loss = 0
    accuracy = 0
    print ("\n\n","*"*5,"Testing the model. Starting", "*"*5 )
    test_result = test_model(model, training_device, testingLoader,test_loss, accuracy,myOptimizer, criterion)
    print ("Testing model succeeded ?", test_result)

    #Clear gpu memory
    torch.cuda.empty_cache()

    #Save the model Checkpoint at this stage
    model.class_to_idx = training_data.class_to_idx
    #get checkpoint file path/name -- this is a REQUIRED CLI arg
    checkpoint_file = in_args.savedir + model.name + "_checkpoint.chk" 
    #Save checkpoint
    checkpoint = {
        'classifier': model.classifier,
        
        'architecture' : model.name,
              
        'class_to_idx': model.class_to_idx,
              
        'model_state_dict': model.state_dict(),
              
        'optimizer_state_dict': myOptimizer.state_dict(),
              
        'epochs': epochs
    }
    #export checkpoint
    torch.save(checkpoint, checkpoint_file)
    print("Checkpoint saved at :",checkpoint_file )

    
# Call to main function to run the program
if __name__ == "__main__":
    main()
