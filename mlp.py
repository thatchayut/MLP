#!/usr/bin/python

def readFile(file):
    doc = open(file, "r")
    for line in doc:
        print(line)

def getInput():
    number_of_features = raw_input("Number of features : ")
    type(number_of_features)
    number_of_layers = raw_input("Number of hidden layers : ")
    type(number_of_layers)
    number_of_nodes = raw_input("Number of nodes in each hidden layer : ")
    type(number_of_nodes)
    check = False
    while (check == False):
        print("Select Activation function : [1]Sigmoid [2]some function")
        function = raw_input("Function number : ")
        type(function)
        if ((function is  "1") or (function is  "2")):
            check = True
        else:
            print("Invalid function number !")
    
    return number_of_features, number_of_layers, number_of_nodes, function

def main():
    number_of_features, number_of_layers, number_of_nodes, function = getInput()
    # readFile("/home/thatchayut/Desktop/MLP/Flood_data_set_edited.txt")

if __name__ == '__main__':
    main()