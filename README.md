# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Before you start
You'll need the popular machine leanring tools installed in your environment to run this project. Run the following commands below

```pip install numpy pandas matplotlib pil```

or if using conda

```conda install numpy pandas matplotlib pil```

# Usage
## Training
Run the file train.py to train the model to use

Basic command : python train.py data_directory
It should print current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
Optional Arguments :
To save a checkpoint set a save directory as such : python train.py data_dir --save_dir save_checkpoint_directory
To select an arcitecture : python train.py data_dir --arch "vgg16"
Setting hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20
Use GPU for training: python train.py data_dir --device gpu
Output: A trained network ready and saved at a checkpoint for prediccting flower images and identifying the species
## Predicting
Use predict.py to predict flower types with probability of that flower. 
Pass in the path to the image you wish to predict as an absolute path /path/to/image.jpg and return the flower name and class probability

Basic usage: python predict.py /path/to/image.jpg checkpoint
Options:
Return top K most likely classes: python predict.py input checkpoint ---top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_To_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

