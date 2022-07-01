import matplotlib.pyplot as plt
def graph_Accuracy(history):
    '''
    input: history of the model's training process
    output: 2 Graphs of the Accuracy: train,validation
    '''
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy') #the title of the graph
    plt.ylabel('accuracy') #the Y in the graph
    plt.xlabel('epoch') #the X in the graph
    plt.legend(['train', 'validation'], loc='upper left') #naming the graphs and putting them in it
    print("a graph of model's Accuracy")
    plt.show() #show graph
    return

def graph_Loss(history):
    '''
    input: history of the model's training process
    output: 2 Graphs of the Loss: train,validation
    '''
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss') #the title of the graph
    plt.ylabel('loss') #the Y in the graph
    plt.xlabel('epoch') #the X in the graph
    plt.legend(['train', 'validation'], loc='upper left') #naming the graphs and putting them in it
    print("a graph of model's loss")
    plt.show() #show graph
    return