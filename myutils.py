# PROGRAMMER: Akah harvey
# DATE CREATED: 14/12/2019
# REVISED DATE: 14/12/2019            <=(Date Revised - if any)
# PURPOSE: utility program for parsing training input arguments, initiatilizing datasets and configuring training models

#utility imports
import argparse
from time import time, sleep
from os import listdir

def get_input_args(prog = 'train'):
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     CLI arguments are expected depending on the calling program
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    args = argparse.ArgumentParser( description='Parse program inputs for this ML flower classification project')
    if prog == 'train':
    #training arguments
        args.add_argument('-a','--arch', default='vgg', type=str, help='The CNN architecture to use for the operation.', metavar='A')
        args.add_argument('-d','--datadir', default='flowers/', type=str, help='The directory path of the flower images to label. Use relative paths', metavar='D')
        args.add_argument('-s','--savedir', default='checkpoints/', type=str, help='The directory path to save checkpoint to. Use relative paths', metavar='C')
        args.add_argument('-r','--learning_rate', default=0.01, type=int, help='The learning rate for the given architecture')
        args.add_argument('-hu','--hidden_units', default=512, type=int)
        args.add_argument('-e','--epochs', default=10, type=int, help='The training epochs to use during the training')
        args.add_argument('-dev','--device', default='cuda', type=str, help='The device to perform training on')
        args.add_argument('-means','--data_means', default='0.485, 0.456, 0.406', type=str, help='The normalisation means for the images')
        args.add_argument('-cnk','--chunks', default=72, type=int, help='The chunks sizes to use for the dataloaders')
        args.add_argument('-stds','--data_stds', default='0.229, 0.224, 0.225', type=str, help='The standard deviations for normalizing images with')
        args.add_argument('output_classes', type=int, action='store', help='MUST - The required number of output classes to be int he output layer') 
    else:
    #predicting arguments
        args.add_argument('input', action="store", help='The input image to predict') #obligatory   
        args.add_argument('checkpoint', action='store', help='The optional checkpoint to restore') #obligatory    
        args.add_argument('--category_names',default='cat_to_name.json', help='The category input files to use') #optional    
        args.add_argument('--top_k',default=3, type=int, help='The number of K most likely classes for the input image') #optional
        args.add_argument('--gpu',default='cpu', help='The platform/processor to use for the training and prediction') #optional    

    return args.parse_args()

def get_model_nodes(model_arch,model_name):
    #in case we might want to have different models, we need to properly setup the right model nodes for that chosen model. Default : vgg16
    m_name=model_name[:3]
    if m_name == 'ale':
        model_nodes = model_arch.classifier[1].in_features
    elif m_name == 'vgg':
        model_nodes = model_arch.classifier[0].in_features
    elif m_name == 'res':
        model_nodes = model_arch.fc.in_features
    elif m_name == 'squ':
        model_nodes = model_arch.classifier[1].in_channels
    else: #default fallback to vgg model
        model_nodes = 25088 # for vgg16 model
    
    return model_nodes
    
def display_results(probs, indexes, classes, flowerList):
    print("="*5, "Prediction results", "="*5)
    print ('\nTop Probabilities: ', probs)
    print('\nTop Model Indexes:', indexes)
    print('\nTop Flower Class Ids:', classes)
    print('\nTop Class Ids with Flower Names:',flowerList)
    

