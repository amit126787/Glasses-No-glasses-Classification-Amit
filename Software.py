import numpy as np
import os
from Image_Selector_and_Prediction import Predict
from modell import train_model
def check_user_answer(user_answer,bool,model,img_height,img_width):
    '''
        input: a user input(y/n), a True bool var, the model var and Height and Width of the images
        output: predict the photo that he will choose
    '''
    while (bool==True):
        while user_answer != 'y' and user_answer != 'n':   #User must input correctly
            user_answer = input("Do you want to make a prediction on an image? (y=yes/n=no) ")
        if user_answer == 'y': #user chooses to make a Prediction
            Predict(model, img_height, img_width) #The Prediction process
            user_answer = 'a' #random character that would activate the choice again
        if user_answer == 'n': #user chooses to end
            bool=False #end the loop

def check_load_answer(load_answer,model,main_dir,train_gen, validation_gen,train_samples, validation_samples,batch_size,epochs):
    '''
        input: load_answer - a user input(y/n), model - the model var, main_dir - main directory path to our project,
         train_gen, validation_gen - 2 'Generator' objects for train and validation, train_samples - number of images in train and in validation,
         batch_size - num of batch size, epochs - number of epochs
        output: history of the model's training process
    '''
    while load_answer != 'y' and load_answer != 'n' :
        load_answer=input("load a pre-trained model , and skip the training process? (y=yes/n=no) ")
    if load_answer == 'y': #user chooses pre-trained model
        model.load_weights(os.path.join(main_dir, 'saved_weights.h5')) #loading the pre-trained model
        history=np.load(os.path.join(main_dir,'saved_history.npy'), allow_pickle='TRUE').item() #get the training history
    if load_answer == 'n': #user chooses to train the model himself
        model, history = train_model(model,train_gen,train_samples,batch_size,epochs,validation_gen,validation_samples) #Model training
        model.save_weights(os.path.join(main_dir, 'saved_weights.h5')) #Save the trained model weights
        np.save(os.path.join(main_dir,'saved_history.npy'), history.history) #Save the history of training
        history=history.history
    return history

def Select_directroy(string):
    '''
    input: one string that is printed
    output: The path of a directory, chosen by the user
    '''
    while(True): #check if path exist
        Directory_Path = input(string)
        if os.path.isdir(Directory_Path) & os.path.exists(Directory_Path):
            return Directory_Path #returns the chosen directory's path
        else:
            print("enter an existed path that is a directory")