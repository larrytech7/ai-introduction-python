# PROGRAMMER: Akah harvey
# DATE CREATED: 14/12/2019
# REVISED DATE: 14/12/2019            <=(Date Revised - if any)
# PURPOSE: Nanodegree program in ML for predicting flower images from a pretrained CNN model 

#imports
from myutils import get_input_args
from myutils import display_results

import os.path
import json
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as Functional
from torchvision import datasets, transforms, models

def predict(file_tensor, model, device, top_k):
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device);
   
    #Set model to evaluate
    model.eval();
    
    #Reshape to [1,3,224,224]
    tensor_image = Functional.interpolate(file_tensor.unsqueeze(0),scale_factor=1)
    #set device
    tensor_image = tensor_image.to(device);

    # TODO: Calculate the class probabilities (softmax) for image
    ps = torch.exp(model(tensor_image))
    top_p, top_index = ps.topk(top_k, dim=1)
    
    # flatten the tensors first 
    top_p_flat = [element.item() for element in top_p.flatten()]
    top_index_flat = [element.item() for element in top_index.flatten()]
    
    # also need the dictionary of indexes -to- image file class codes 
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    #Then check for a cat_to_name json file
    catfile = True if in_args.category_names else False
    
    #Get list of class_ids
    class_ids = list()
    for idx in top_index_flat:
        class_ids.append(idx_to_class[idx])
    
    #now want to build the images names if a cat_to_name json file was provided
    top_classes = {}
    if catfile:
        with open(in_args.category_names, 'r') as f:
            category_names = json.load(f)
        for clsstr in class_ids:
            top_classes.update({clsstr: category_names[str(clsstr)]})
                
    #returning probabilities as Lists according to the instructions, and Class List + flowers as a dictionary
    return ["{:.2%}".format(x) for x in top_p_flat], top_index_flat, class_ids, top_classes

def load_checkpoint(checkpoint_path):
    #load in checkpoint data
    checkpoint = torch.load(checkpoint_path)
    
    model_name = checkpoint['architecture']
    #set our model type
    str = "models." + model_name + "(pretrained=True)"

    model = eval(str)
    for param in model.parameters():
        param.requires_grad = False
    
    #load other params into model saved at checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])

    #for other items in the checkpoint- like epochs and optimizer state
    #these can be addressed once model returned 
    
    model.model_name = model_name
    return model

def open_file(mFile):
    rt_tensor=torch.tensor
    
    try:
        infer_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        pil_image = Image.open(mFile)
        img_tensor = infer_transform(pil_image)
        rt_tensor =  img_tensor
    except:
        print ("An error occurred while attempting to open file. {}".format(mFile), sys.exc_info()[0])
        
    return rt_tensor

#get program  input
in_args = get_input_args()

#load main program
def main():
    #load input file
    input_file = in_args.input

    if os.path.isfile(input_file):
        file_tensor = open_file(input_file)
    else:# file can't be ound we need to stop
        print("Can't find image to predict. Check that this file exists and it's name is correctly passed.")
        exit
        
    #Load the model from checkpoint file
    if os.path.isfile(in_args.checkpoint):
        checkpoint_file = in_args.checkpoint
    else:#invalid checkpoint passed
        print("Can't find model checkpoint file provided. Please check that this is a valid checkpoint file and try again")
        exit
    # load a checkpoint
    test_model = load_checkpoint(checkpoint_file)
    print("Using Torchvision model:",test_model.model_name)

    # Set device to use to default to cpu
    test_device = torch.device(in_args.gpu)
    device = in_args.gpu
    if device != None: 
        if device == "cuda":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                test_device = torch.device('cuda')
            else:
                test_device = torch.device('cpu')

    print("Using : {}".format(test_device), "for training")

    # Get top_k argument
    tLambda = lambda x: 5 if x == None else x
    top_k = tLambda(in_args.top_k)

    print("Listening to {} possibilities ".format(top_k))

    # run prediction and return outputs
    probabilities, mIndexes, classes, flowerlist = predict(file_tensor, testing_model, test_device, top_k)

    #Display prediction results
    display_results(probabilities, mIndexes, classes, flowerlist)
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()