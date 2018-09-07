#!/usr/bin/python
import numpy as np
import pandas

def readFile(file):
    doc = open(file, "r")
    for line in doc:
        print(line)

def getInput():
    number_of_features = input("Number of features : ")
    type(number_of_features)
    number_of_layers = input("Number of hidden layers : ")
    type(number_of_layers)
    number_of_nodes = input("Number of nodes in each hidden layer : ")
    type(number_of_nodes)
    number_of_classes = input("Number of output classes : ")
    type(number_of_classes)
    check = False
    while (check == False):
        print("Select Activation function : [1]Sigmoid [2]some function")
        function = input("Function number : ")
        type(function)
        if ((function is  "1") or (function is  "2")):
            check = True
        else:
            print("Invalid function number !") 
    number_of_classes = input("Learning rate : ")
    type(learning_rate)
    number_of_classes = input("Momentum : ")
    type(momentum) 
    return number_of_features, number_of_layers, number_of_nodes, function, number_of_classes,
           learning_rate, momentum


def createInputNodes(number_of_features):
    arr_input_nodes = np.zeros(int(number_of_features))
    return arr_input_nodes

def createHiddenLayers(number_of_features,number_of_layers,number_of_nodes,number_of_classes):
    # first hidden layer that is connected with input nodes
    first_layer = []
    last_layer = []
    node = []
    layer = []
    final = []
    count = 0
    while count < int(number_of_nodes):
        arr = np.random.uniform(low=-1.0,high=1.0,size=int(number_of_features))
        first_layer.append(arr)
        count += 1
    final.append(first_layer)
    # the rest hidden layers
    # create all hidden layers except the first and the last layer that
    # are connected to input nodes and output nodes respectively
    for layer_count in range(0, int(number_of_layers)-2):
        for node_count in range(0,int(number_of_nodes)):
            arr = np.random.uniform(low=-1.0,high=1.0,size=int(number_of_nodes))
            node.append(arr)
        layer.append(node)
    final.append(layer)
    #The last hidden layer connected to an output layer
    count = 0
    while count < int(number_of_nodes):
        arr = np.random.uniform(low=-1.0,high=1.0,size=int(number_of_classes))
        last_layer.append(arr)
        count += 1
    final.append(last_layer)
    print(final)
def main():
    number_of_features, number_of_layers, number_of_nodes, function, number_of_classes = getInput()
    arr_input_nodes = createInputNodes(number_of_features)
    createHiddenLayers(number_of_features,number_of_layers,number_of_nodes,number_of_classes) 
    # readFile("/home/thatchayut/Desktop/MLP/Flood_data_set_edited.txt")

if __name__ == '__main__':
    main()