from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
def Select_Image(string):
    '''
    input: one string that is printed
    output: the path of an image, writen by the user
    '''
    while (True):  # check if path exist
        filepath = input(string)
        if os.path.isfile(filepath) & os.path.exists(filepath):
            return filepath  # returns the chosen directory's path
        else:
            print("enter an existed path that is an image")

def Predict(model, img_height, img_width):
    '''
    input: the model, image height, image width
    output: a window with the selected image and the class of that image (predicted by the model)
    '''
    string = "write an image path to test (PNG/JPEG ONLY)" #the string that is printed for the user
    img_path = Select_Image(string) #The Choosing of the File
    img= image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) #convert the image to an array
    img_array /= 255.0 #the model can only work with pixel values between 0 and 1, so we rescale
    img_batch = np.expand_dims(img_array, axis=0)
    plt.imshow(img)
    #prediction= model.predict_classes(img_batch)
    prediction = (model.predict(img_batch) > 0.5).astype("int32")
    if (prediction==0): #predction = 0 class is Glasses
        plt.title('Glasses') #title the image Glasses
    else:
        plt.title('No Glasses') #title the image No Glasses
    plt.show() #shows the image with the correct title
    return