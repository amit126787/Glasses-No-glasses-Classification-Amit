from zipfile import ZipFile


def open_zip(file_name):
    '''
    input: the file name
    output: print all the files that have been extracted from the zip file
    '''
    # opening the zip file in READ mode
    with ZipFile(file_name, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()

        # extracting all the files
        print('Extracting all the files now...')
        zip.extractall()
        print('Done!')