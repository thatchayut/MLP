#!/usr/bin/python
import numpy as np
import pandas
import init_node as init

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
        print("Select Activation function : [1]Sigmoid [2]Hyperbolic Tangent")
        function = input("Function number : ")
        type(function)
        if ((function is  "1") or (function is  "2") or (function is "3") or (function is "4")):
            # Check for additional arguments
            if ((function is "3") or (function is "4")):
                beta = input("bata value : ")
                type(beta)
                while(beta.isnumeric() == False):
                    beta = input("bata value : ")
                    type(beta)
                check = True
            else:
                beta = None
                check = True          
        else:
            print("Invalid function number !") 
    learning_rate = input("Learning rate : ")
    type(learning_rate)
    momentum = input("Momentum : ")
    type(momentum) 
    return number_of_features, number_of_layers, number_of_nodes, function, number_of_classes, \
           learning_rate, momentum, beta

def main():
    #initialize network
    number_of_features, number_of_layers, number_of_nodes, function, number_of_classes, learning_rate, momentum, beta = getInput()
    arr_input_nodes = init.createInputNodes(number_of_features)
    arr_hidden_layers = init.createHiddenLayers(number_of_features,number_of_layers,number_of_nodes,number_of_classes) 
    arr_weight_bias, arr_bias = init.createBias(number_of_nodes,number_of_layers)
    arr_output_nodes = init.createOutputNodes(number_of_classes)


if __name__ == '__main__':
    main()