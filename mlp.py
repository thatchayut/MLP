#!/usr/bin/python
import numpy as np
import init_node as init
import cross_validation as cv
import math

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
        if ((function is  "1") or (function is  "2")):
            # Check for additional arguments
            # if ((function is "3") or (function is "4")):
            #     beta = input("bata value : ")
            #     type(beta)
            #     while(beta.isnumeric() == False):
            #         beta = input("bata value : ")
            #         type(beta)
            #     check = True
            # else:
            beta = None
            check = True          
        else:
            print("Invalid function number !") 
    learning_rate = input("Learning rate : ")
    type(learning_rate)
    momentum = input("Momentum : ")
    type(momentum) 
    check = False
    print("k-fold cross validation ...")
    print("Enter k ...")
    while (check == False):
        fold = input("k : ")
        type(fold)
        if((fold.isnumeric() == True) and (int(fold) > 1)):
            check = True
        else:
            print("Invalid input ...")
    return number_of_features, number_of_layers, number_of_nodes, function, number_of_classes, \
           learning_rate, momentum, beta, fold

def main():
    #initialize network
    number_of_features, number_of_layers, number_of_nodes, function, number_of_classes, learning_rate, momentum, beta, fold = getInput()
    arr_input_nodes = init.createInputNodes(number_of_features)
    arr_hidden_layers = init.createHiddenLayers(number_of_features,number_of_layers,number_of_nodes,number_of_classes) 
    arr_hidden_layers_new = init.createHiddenLayers(number_of_features,number_of_layers,number_of_nodes,number_of_classes) 
    arr_Y = init.createY(number_of_nodes, number_of_layers)
    arr_weight_bias, arr_bias = init.createBias(number_of_nodes, number_of_layers)
    arr_output_nodes = init.createOutputNodes(number_of_classes)
    arr_weight_bias_output, arr_bias_output  =init.createBias(number_of_classes, 1)
    arr_grad_output = init.createLocalGradOutput(number_of_classes)
    arr_grad_hidden = init.createLocalGradHidden(number_of_nodes, number_of_layers)
    cv.crossValidation("flood-input.csv", "flood-output.csv", "flood-data-full.csv", fold, arr_input_nodes, arr_hidden_layers, arr_hidden_layers_new,  arr_Y, arr_output_nodes, arr_weight_bias, arr_bias, \
                        arr_weight_bias_output, arr_bias_output, function, momentum, learning_rate, beta, arr_grad_hidden, arr_grad_output)
    print("arr_hidden_layers : ")
    print(arr_hidden_layers)
    print("arr_hidden_layers_new : ")
    print(arr_hidden_layers)
    print()
    print("size of list containing hidden layer : " + str(len(arr_hidden_layers)))
    print(str(len(arr_hidden_layers[1])) + " layer(s) of weigh connected to hidden node")
    print("1 layer of weight connected to INPUT layer")
    print("1 layer connected to OUTPUT layer")
    print("total layer of weight : " + str(1 + len(arr_hidden_layers)))
    #FOR DEBUGGING!!!
    # print("all layer : " + str(len(arr_hidden_layers)))
    # print("hidden : " + str(len(arr_hidden_layers[1])))
    # print("member in hidden : " + str(len(arr_hidden_layers[1][0])))
    # print(arr_weight_bias)
    # print(arr_Y)
    print("arr_weight_bias_output : " + str(arr_weight_bias_output))
    print("arr_bias_output : " + str(arr_bias_output))
    print("arr_weight_bias : " + str(arr_weight_bias))
    print("arr_bias : " + str(arr_bias))
    print("arr_grad_output : " + str(arr_grad_output))
    print("arr_grad_hidden : " + str(arr_grad_hidden))
    print()


if __name__ == '__main__':
    main()