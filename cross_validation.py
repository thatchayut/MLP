import pandas
import numpy as np
import random
import math
import init_node as init

def readFile(file):
    data = pandas.read_csv(file)
    dataframe = pandas.DataFrame(data)
    number_of_data = dataframe.shape[0] + 1
    print("Number of data : " + str(number_of_data))
    # array contains indicators of each data
    arr_row = np.arange(1,number_of_data + 1)
    # print(arr_row)
    # shuffle to decrease order dependency
    random.shuffle(arr_row)
    return data, dataframe, number_of_data, arr_row

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def forward (dataframe_input, dataframe_output, line, arr_input_nodes, arr_output_nodes, arr_Y, arr_hidden_layers, \
            number_of_nodes, number_of_layers):
    # change number of line in to dataframe
    line = line - 2
    # print("line : " + str(line + 2))
    print(dataframe_input.iloc[line])
    data_input = dataframe_input.iloc[line]
    data_output = dataframe_output.iloc[line]
    print(data_output)
    print(data_input.shape[0])

    # check if input node is enough
    # data.shape[0] - 1 = actual inputs (desired output is excluded)
    # assign value to input nodes
    check = False
    if ((data_input.shape[0]) == len(arr_input_nodes)):
        print("OK")
        check = True
    else:
        print("invalid input nodes")
        print()
    
    count = 0
    if (check == True):
        for data_element in data_input:
            arr_input_nodes[count] = data_element
            count += 1
        print("input : " + str(arr_input_nodes))
        print()
    
    # assign value to output nodes
    check = False
    if ((data_output.shape[0]) == len(arr_output_nodes)):
        print("OK")
        check = True
    else:
        print("invalid output nodes")
        print()
    
    count = 0
    if (check == True):
        for data_element in data_output:
            arr_output_nodes[count] = data_element
            count += 1
        print("output : " + str(arr_output_nodes))
        print()        
    
    # CALCULATE Y of each node 
    for layer_index in range(0, len(arr_Y)):
        if (layer_index == 0):
            for index in range(0, len(arr_Y[layer_index])):
                for node_index in range(0, len(arr_hidden_layers[layer_index])):
                    for weight in arr_hidden_layers[layer_index][node_index]:
                        arr_Y[layer_index][index] += (weight * arr_input_nodes[node_index])
                    # arr_Y[layer_index][index] += (arr_hidden_layers[layer_index][node_index] * arr_input_nodes[node_index])
    print("arr_Y" + str(arr_Y))

    #reset arr_Y
    for layer_index in range(0, len(arr_Y)):
        for node_index in range(0,len(arr_Y[layer_index])):
            arr_Y[layer_index][node_index] = 0
    print("arr_Y after reset: " + str(arr_Y))

def crossValidation(input_file, output_file, number_of_fold, arr_input_nodes, arr_hidden_layers, arr_Y, arr_output_nodes, arr_weight_bias, arr_bias, \
                    function, momentum, learning_rate, beta, number_of_nodes, number_of_layers):
    data_input, dataframe_input, number_of_data_input, arr_row_input = readFile(input_file)
    data_output, dataframe_output, number_of_data_output, arr_row_output = readFile(output_file)
    print(dataframe_output)
    # dataframe = pandas.DataFrame(data)
    # # print(dataframe.iloc[0])
    # number_of_data = dataframe.shape[0] + 1
    # print("Number of data : " + str(number_of_data))
    # # array contains indicators of each data
    # arr_row = np.arange(1,number_of_data + 1)
    # # print(arr_row)
    # # shuffle to decrease order dependency
    # random.shuffle(arr_row)
    # print(arr_row)
    # find size of each part
    # number_of_fold = 5 # JUST FOR TEST!!!
    size = math.ceil(number_of_data_input/int(number_of_fold))
    # print(size)
    # split data into k parts
    data_chunk_input = list(chunks(arr_row_input, size))
    print("\nData chunks ...")
    print(data_chunk_input)
    # print (len(data_chunk))
    # test and train
    count = 0
    for test_element in data_chunk_input:
        count += 1
        print("------------------------------" + str(count) + " fold ------------------------------")
        test_part = test_element
        for train_element in data_chunk_input:
            if(train_element not in test_part):
                print("TRAIN----------------")
                print(train_element)
                print()
                for element in train_element:
                    forward(dataframe_input, dataframe_output, element, arr_input_nodes, arr_output_nodes, arr_Y, \
                    arr_hidden_layers, number_of_nodes, number_of_layers)
        print("TEST------")
        print(test_part)
        print()