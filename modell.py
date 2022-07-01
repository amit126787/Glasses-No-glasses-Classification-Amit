from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
def create_model(img_height, img_width):
    '''
    Input: Height and Width of our images (after resize)
    Output: The model (Object keras.Model)
    '''
    # checking the format of the images, for setting a correct input shape
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_height, img_width)
    else:
        input_shape = (img_height, img_width, 3)
    # Creating the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  # use the loss function "Binary Cross Entropy to evaluate the model.
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())  # summary of the model
    return model

def train_model(model,train_generator, train_samples,batch_size,epochs,validation_generator, validation_samples):
    '''
    Input: the model (Object keras.Model)
    Output: A trained model, and the history - for plotting purposes
    '''
    earlyStopping=tf.keras.callbacks.EarlyStopping(patience=2) #Model Training involves Early Stopping - prevents overfitting
    history= model.fit_generator(train_generator,
    steps_per_epoch = train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    verbose=2,
    callbacks=[earlyStopping])
    return model, history

def Preprocessing(img_height, img_width,train_dir, validation_dir,batch_size):
    '''
    input: data_dir = path of directory which contains 2 sub-directories for train and validation,
    img_height, img_width = the wanted size of the images,
    train_dir, validation_dir = the 2 sub-directories, each containing 2 sub-directories for glasses and no-glasses images
    batch_size = the batch size that we'll use
    output: 2 'Generator' objects for train and validation that we'll use in the training of the model
    '''
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    validation_datagen=ImageDataGenerator(rescale=1./255)
#creating a generator for train data
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=(img_height, img_width),  # images resized
        batch_size=batch_size,
        class_mode='binary'  #binary labels
        )

# this is a similar generator, for validation data
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary'
            )
    return train_generator, validation_generator