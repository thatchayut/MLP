import pandas
import numpy as np
import random
import math
import init_node as init
import function
import statistics

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

def featureScaling(input_data, output_data, min_value, max_value, mean_value):
    normalized_input_data = []
    for element in input_data:
        # data[index] = (data[index] - mean_value)/(max_value - min_value)
        result = (element - min_value)/(max_value - min_value)
        result = round(result,5)
        normalized_input_data.append(result)
        # print("testtttttttttt")
    print("normalized_input_data : " + str(normalized_input_data))
    normalized_output_data = []
    for element in output_data:
        # data[index] = (data[index] - mean_value)/(max_value - min_value)
        result = (element - mean_value)/(max_value - min_value)
        result = round(result,5)
        normalized_output_data.append(result)
        # print("testtttttttttt")
    print("normalized_output_data : " + str(normalized_output_data))

    return normalized_input_data, normalized_output_data

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def useFunction(data, function_number, beta):
    if (function_number == "1"):
        # print("sigmoiddddddddd")
        return function.sigmoid(data)  
    elif(function_number == "2"):
        # print("hyperrrrrrrrr")
        return function.hyperbolicTangent(data)      
    elif(function_number == "3"):
        # print("unittttttttt")
        return function.unitStep(data, beta)
    elif(function_number == "4"):
        # print("rampppppp")
        return function.sigmoid(data, beta)  

def calculateError(actual_output, desired_output):
    arr_error = []
    sse = 0
    for index in range(0, len(actual_output)):
        error_value = (desired_output[index] - actual_output[index])
        error_percentage = ((error_value/actual_output[index]) * 100)
        arr_error.append(error_value)
        print("absolute error of output #" + str(index) + " = " + str(error_value) + "(" + str(error_percentage) + " %)")
    print("arr _error : " + str(arr_error))
    # calculate Sum Square Error (SSE)
    for element in arr_error:
        sse += (1/2)*(element * element)
    print("sse : " + str(sse))
    return sse, arr_error

def forward (dataframe_input, dataframe_output, data_all, line, arr_input_nodes, arr_output_nodes, arr_Y, arr_hidden_layers,\
            arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, beta):
    # calculate min, max, mean to be used in feature scaling
    min_value = data_all.min()
    max_value = data_all.max()
    mean_value = data_all.mean()
    mean_value = round(mean_value, 5)
    print("MIN from ALL : " + str(min_value[1]))
    print("MAX from ALL : " + str(max_value[1]))
    print("MEAN from ALL : " + str(mean_value[1]))
    # change number of line in to dataframe
    line = line - 2
    # print("line : " + str(line + 2))
    data_input = dataframe_input.iloc[line]
    print(data_input)
    # data_input = featureScaling(data_input)
    # print(data_input)
    data_output = dataframe_output.iloc[line]
    print(data_output)
    # data_output = featureScaling(data_output)
    data_input, data_output = featureScaling(data_input, data_output, min_value[1], max_value[1], mean_value[1])
    # print(len(data_input))

    # check if input nodes are enough
    input_check = False
    if (len(data_input) == len(arr_input_nodes)):
        print("Input later : OK")
        print()
        input_check = True
    else:
        print("invalid input nodes")
        print() 
     # assign value to input nodes  
    count = 0
    if (input_check == True):
        for data_element in data_input:
            arr_input_nodes[count] = data_element
            count += 1
        print("input : " + str(arr_input_nodes))
        # print()
    
    # check if output nodes are enough
    output_check = False
    if (len(data_output) == len(arr_output_nodes)):
        print("Output layer : OK")
        print()
        output_check = True
    else:
        print("invalid output nodes")
        print()
    
    # count = 0
    # if (check == True):
    #     for data_element in data_output:
    #         arr_output_nodes[count] = data_element
    #         count += 1
    #     print("output : " + str(arr_output_nodes))
    #     print()        
    #     print("arr_output_nodes : " + str(arr_output_nodes))
    #     print()
        # CALCULATE Y of each node only when INPUT and OUTPUT are VALID
    if ((input_check == True) and (output_check == True)):
        print("BEFORE... output nodes : " + str(arr_output_nodes))
        for layer_index in range(0, len(arr_Y)):
            # weight from an input layer to the 1st hidden layer
            if (layer_index == 0):
                for index in range(0, len(arr_Y[layer_index])):
                    for node_index in range(0, len(arr_hidden_layers[0])):
                        for weight in arr_hidden_layers[0][node_index]:
                            arr_Y[layer_index][index] += (weight * arr_input_nodes[node_index])
                            arr_Y[layer_index][index] += (arr_weight_bias[layer_index][index] * arr_bias[layer_index][index])
                    # modify output using activation function
                    arr_Y[layer_index][index] = useFunction(arr_Y[layer_index][index], function_number, beta)
            # calcualte output at the last hidden layer
            elif (layer_index == (len(arr_Y) - 1)):
                # print("testtttttttttttttttttttttt")
                for output_index in range(0, len(arr_output_nodes)):
                    for index in range(0, len(arr_Y[layer_index])):
                        for node_index in range(0, len(arr_hidden_layers[2])):
                            for weight in arr_hidden_layers[2][node_index]:
                                arr_output_nodes[output_index] += (weight * arr_Y[layer_index][index])
                    arr_output_nodes[output_index] += (arr_weight_bias_output[output_index] * arr_bias_output[output_index])
                    arr_output_nodes[output_index] = useFunction(arr_output_nodes[output_index], function_number, beta) 
            else:
            # calculate output for all nodes in hidden layers except the first layer connected to input node
                for layer_index in range(1, len(arr_Y)):
                    for index in range(0, len(arr_Y[layer_index])):
                        for weight_layer in range(0, len(arr_hidden_layers[1])):
                            for node_index in range(0, len(arr_hidden_layers[1][weight_layer])):
                                for weigth_to_node in range(0, len(arr_hidden_layers[1][weight_layer][node_index])):
                                    arr_Y[layer_index][index] += (arr_Y[layer_index - 1][node_index] * arr_hidden_layers[1][weight_layer][node_index][weigth_to_node])
                                    arr_Y[layer_index][index] += (arr_weight_bias[layer_index][index] * arr_bias[layer_index][index])
                    arr_Y[layer_index][index] = useFunction(arr_Y[layer_index][index], function_number, beta)
        print("arr_Y" + str(arr_Y))
        print("arr_output_nodes(actual output) : " + str(arr_output_nodes))
        print("data output(desired output)  : " + str(data_output))
        sse, arr_error = calculateError(arr_output_nodes, data_output)
    else:
        print("cannot do FORWARDING!")
        print()
    return sse, arr_error
    # #reset arr_Y
    # for layer_index in range(0, len(arr_Y)):
    #     for node_index in range(0,len(arr_Y[layer_index])):
    #         arr_Y[layer_index][node_index] = 0
    # print("arr_Y after reset: " + str(arr_Y))

    # #reset arr_output_nodes
    # for node_index in range(0, len(arr_output_nodes)):
    #     arr_output_nodes[node_index] = 0
    # print("arr_output_nodes after reset : " + str(arr_output_nodes))
    # print("------------------------------------------------------------------------------------------------------")

def backward(arr_grad_hidden, arr_grad_output, arr_Y, arr_output_nodes, arr_error):
    new_arr = []
    new_arr.append(arr_Y)
    new_arr.append(arr_output_nodes)
    arr_grad = []
    arr_grad.append(arr_grad_hidden)
    arr_grad.append(arr_grad_output)
    print("arr_Y : " + str(arr_Y))
    print("new_arr" + str(new_arr))
    print("arr_grad_hidden, arr_grad_output" + str(arr_grad))

def crossValidation(input_file, output_file, full_data_file, number_of_fold, arr_input_nodes, arr_hidden_layers, arr_Y, arr_output_nodes, arr_weight_bias, arr_bias, \
                    arr_weight_bias_output, arr_bias_output, function_number, momentum, learning_rate, beta, arr_grad_hidden, arr_grad_output):
    data_input, dataframe_input, number_of_data_input, arr_row_input = readFile(input_file)
    data_output, dataframe_output, number_of_data_output, arr_row_output = readFile(output_file)
    data_all, dataframe_all, number_of_data_all, arr_row_all = readFile(full_data_file)
    print(dataframe_output)
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

                    sse, arr_error = forward(dataframe_input, dataframe_output, data_all, element, arr_input_nodes, arr_output_nodes, arr_Y, \
                    arr_hidden_layers, arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, beta)

                    backward(arr_grad_hidden, arr_grad_output, arr_Y, arr_output_nodes, arr_error)

                    #reset arr_Y
                    for layer_index in range(0, len(arr_Y)):
                        for node_index in range(0,len(arr_Y[layer_index])):
                            arr_Y[layer_index][node_index] = 0
                    print("arr_Y after reset: " + str(arr_Y))

                    #reset arr_output_nodes
                    for node_index in range(0, len(arr_output_nodes)):
                        arr_output_nodes[node_index] = 0
                    print("arr_output_nodes after reset : " + str(arr_output_nodes))
                    print("------------------------------------------------------------------------------------------------------")
        print("TEST------")
        print(test_part)
        print()