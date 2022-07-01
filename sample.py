import os
def sample(train_dir, validation_dir):
    '''
    Input: path of train and validation directories
    Output: number of images in train and in validation (seperate variables)
    '''
    #putting in variables the directory paths for train class-subfolders
    glasses_train_dir = os.path.join(train_dir, 'glasses')
    no_glasses_train_dir = os.path.join(validation_dir,'no_glasses')
    #putting in variables the directory paths for validation class-subfolders
    glasses_validation_dir=os.path.join(validation_dir, 'glasses')
    no_glasses_validation_dir=os.path.join(validation_dir, 'no_glasses')
    #vars amount of images in train and validation
    train_samples = len(os.listdir(glasses_train_dir))
    train_samples = train_samples + len(os.listdir(no_glasses_train_dir))
    validation_samples = len(os.listdir(glasses_validation_dir))
    validation_samples = validation_samples + len(os.listdir(no_glasses_validation_dir))
    return train_samples, validation_samples