import numpy as np
import os
import warnings
from modell import Preprocessing
from sample import sample
from modell import create_model
from Image_Selector_and_Prediction import Select_Image
from graph_Accuracy_Loss import graph_Accuracy
from graph_Accuracy_Loss import graph_Loss
from zip_extract import open_zip
from Software import check_user_answer
from Software import check_load_answer
from Software import Select_directroy
from sklearn.model_selection import train_test_split
import cv2
import vars


def main():
    warnings.filterwarnings('ignore')
    #Part 1
    #creating vars
    img_height = vars.img_height
    img_width = vars.img_width
    batch_size = vars.batch_size
    epochs = vars.epochs

    answer = input("Do you want to extract files? (y/n)")
    if answer == "y":
        open_zip(input("input zip path"))

    #building directories
    main_dir = Select_directroy("write the folder path in which the project is located ('Glass_No_glasses_project' folder)")
    data_dir = os.path.join(main_dir, 'dataset')
    validation_dir = os.path.join(data_dir, 'validation')
    train_dir = os.path.join(data_dir, 'train')
    
    train_gen, validation_gen = Preprocessing(img_height, img_width, train_dir, validation_dir,batch_size)
    
    #Part 2
    train_samples, validation_samples = sample(train_dir, validation_dir)
    model = create_model(img_height, img_width)
    
    #The process of training, user can choose to train the model himself or use a pre-trained model
    #The only difference is the time it takes
    load_answer=input("load a pre-trained model, and skip the training process? (y=yes/n=no) ")
    history = check_load_answer(load_answer,model,main_dir,train_gen, validation_gen,train_samples, validation_samples,batch_size,epochs)
    
    #Part 3
    #The process of predicting images, user can choose to predict more and more images, or to end
    bool=True #for the end loop
    user_answer = input("Do you want to make a prediction on an image? (y=yes/n=no) ")
    check_user_answer(user_answer,bool,model,img_height,img_width)

    #show 2 images of graphs of the loss and accuracy throughout the model's training
    #"""
    graph_Accuracy(history)
    graph_Loss(history)
    #"""

if __name__=='__main__':
    main()