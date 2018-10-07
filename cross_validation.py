import pandas
import numpy as np
import random
import math
import function
import statistics
import init_node as init
import copy


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

def featureScaling(input_data, output_data):
    normalized_input_data = []
    merged_data = [] 
    merged_data.extend(input_data)
    merged_data.extend(output_data)
    min_value = min(merged_data)
    max_value = max(merged_data)
    for element in input_data:
        result = (element - min_value)/(max_value - min_value)
        result = round(result,5)
        normalized_input_data.append(result)
    # print("normalized_input_data : " + str(normalized_input_data))
    normalized_output_data = []
    for element in output_data:
        result = (element - min_value)/(max_value - min_value)
        result = round(result,5)
        normalized_output_data.append(result)

    return normalized_input_data, normalized_output_data

def convertBack(value, input_data, output_data):
    merged_data = [] 
    merged_data.extend(input_data)
    merged_data.extend(output_data)
    min_value = min(merged_data)
    max_value = max(merged_data)

    new_value = (value * (max_value - min_value)) + min_value
    return new_value

def normalizeError(value, input_data, output_data):
    merged_data = [] 
    merged_data.extend(input_data)
    merged_data.extend(output_data)
    min_value = min(merged_data)
    max_value = max(merged_data)

    new_value = (value - min_value)/(max_value - min_value)
    return new_value

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def useFunction(data, function_number, beta):
    if (function_number == "1"):
        return function.sigmoid(data)  
    elif(function_number == "2"):
        return function.hyperbolicTangent(data)      
    elif(function_number == "3"):
        return function.unitStep(data, beta)
    elif(function_number == "4"):
        return function.sigmoid(data, beta)  

def calculateError(actual_output, desired_output):
    arr_error = []
    sse = 0
    for index in range(0, len(actual_output)):
        error_value = (desired_output[index] - actual_output[index])
        arr_error.append(error_value)
    # calculate Sum Square Error (SSE)
    for element in arr_error:
        sse += (1/2)*(element * element)
    return sse, arr_error

def calcualteMSE(arr_error, size):
    result = 0
    for element in arr_error:
        result += element
    result = result/size
    return result

def forward (dataframe_input, dataframe_output, data_all, line, arr_input_nodes, arr_output_nodes, arr_Y, arr_hidden_layers,\
            arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, beta, number_of_classes):
    # change number of line in to dataframe
    line = line - 2
    data_input = dataframe_input.iloc[line]
    data_input_template = copy.deepcopy(data_input)
    data_output = dataframe_output.iloc[line]
    data_output_template = copy.deepcopy(data_output)

    check_input = True
    check_output = True
    for element in data_input:
        if((element < -1) or (element  > 1)):
            check_input = False
            break
    for element in data_output:
        if((element < -1) or (element  > 1)):
            check_output = False
            break   

    if((check_input == False) and (check_output == False)):
        data_input, data_output = featureScaling(data_input, data_output)

    # check if input nodes are enough
    input_check = False
    if (len(data_input) == len(arr_input_nodes)):
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
    
    # check if output nodes are enough
    output_check = False
    if (len(data_output) == len(arr_output_nodes)):
        output_check = True
    else:
        print("invalid output nodes")
        print()
    
    # CALCULATE Y of each node only when INPUT and OUTPUT are VALID
    if ((input_check == True) and (output_check == True)):
        for layer_index in range(0, len(arr_Y) + 1):
            # calculate output
            if(layer_index == (len(arr_Y))):
                if(number_of_classes == "1"):
                    for output_index in range(0, len(arr_output_nodes)):
                        for weight_node_index in range(0, len(arr_hidden_layers[2])):
                            result = 0
                            result += (arr_hidden_layers[2][weight_node_index] * arr_Y[len(arr_Y) - 1][weight_node_index])
                            result += (arr_weight_bias_output[output_index] * arr_bias_output[output_index])
                        arr_output_nodes[output_index] = result
                        arr_output_nodes[output_index] = useFunction(arr_output_nodes[output_index], function_number, beta)
                else:
                    for output_index in range(0, len(arr_output_nodes)):
                        for weight_node_index in range(0, len(arr_hidden_layers[2])):
                            result = 0
                            for weight_to_node_index in range(0, len(arr_hidden_layers[2][weight_node_index])):
                                result += (arr_hidden_layers[2][weight_node_index][weight_to_node_index] * arr_Y[len(arr_Y) - 1][weight_node_index])
                                result += (arr_weight_bias_output[0][output_index]* arr_bias_output[0][output_index])
                            arr_output_nodes[output_index] = result
                            arr_output_nodes[output_index] = useFunction(arr_output_nodes[output_index], function_number, beta)
            # y at the first hidden layer
            elif(layer_index == 0):
                for weight_node_index in range(0, len(arr_hidden_layers[0])):
                    result = 0
                    if(number_of_classes == "1"):
                        for weight_to_node_index in range(0, len(arr_hidden_layers[0][weight_node_index])):
                            result += (arr_input_nodes[weight_node_index] * arr_hidden_layers[0][weight_node_index][weight_to_node_index])
                    else:
                         for arr_input_index in range(0, len(arr_input_nodes)):
                            for weight_to_node_index in range(0, len(arr_hidden_layers[0][weight_node_index])):
                                result += (arr_input_nodes[arr_input_index] * arr_hidden_layers[0][weight_node_index][weight_to_node_index])
                    result += (arr_bias[0][weight_node_index] * arr_weight_bias[0][weight_node_index])
                    arr_Y[0][weight_node_index] = result
                    arr_Y[0][weight_node_index] = useFunction(arr_Y[0][weight_node_index], function_number, beta)
            # y at all hidden layers except the first layer
            else:
                for arr_Y_layer_index in range(1, len(arr_Y)):
                    for arr_Y_node_index in range(0, len(arr_Y[arr_Y_layer_index])):
                        for weight_layer_index in range(0, len(arr_hidden_layers[1])):
                            # only use a layer that is macheed with arr_Y_layer_index
                            if(weight_layer_index == (arr_Y_layer_index - 1)):
                                for weight_node_index in range(0, len(arr_hidden_layers[1][weight_layer_index])):
                                    if(arr_Y_node_index == weight_node_index):
                                        result = 0
                                        for weight_to_node_index in range(0, len(arr_hidden_layers[1][weight_layer_index][weight_node_index])):
                                            result == (arr_hidden_layers[1][weight_layer_index][weight_node_index][weight_to_node_index] * \
                                                        arr_Y[arr_Y_layer_index - 1][weight_to_node_index])
                                        result += (arr_bias[weight_layer_index][arr_Y_node_index] * arr_weight_bias[weight_layer_index][arr_Y_node_index])
                                arr_Y[arr_Y_layer_index][arr_Y_node_index] = result
                                arr_Y[arr_Y_layer_index][arr_Y_node_index] = useFunction(arr_Y[arr_Y_layer_index][arr_Y_node_index], function_number, beta)

        if((check_input == False) and (check_output == False)):
            converted_arr_output_node = []
            for element_index in range(0, len(arr_output_nodes)):
                converted_value = convertBack(arr_output_nodes[element_index], data_input_template, data_output_template)
                converted_arr_output_node.append(converted_value)

            sse, arr_error = calculateError(converted_arr_output_node, data_output_template)
            predicted_output = copy.deepcopy(converted_arr_output_node)
            converted_arr_output_node.clear()
            
            #normalize error
            normalized_arr_error = []
            for element_index in range(0, len(arr_error)):
                error = normalizeError(arr_error[element_index], data_input_template, data_output_template)
                normalized_arr_error.append(error)

            return arr_input_nodes, sse, normalized_arr_error, predicted_output, data_output_template
        else:
            sse, arr_error = calculateError(arr_output_nodes, data_output_template)
            predicted_output = copy.deepcopy(arr_output_nodes)
            return arr_input_nodes, sse, arr_error, predicted_output, data_output_template
    else:
        print("cannot do FORWARDING!")
        print()

def backward(arr_input_nodes_with_value, arr_hidden_layers, arr_hidden_layers_new, arr_grad_hidden, arr_grad_output, arr_Y, arr_output_nodes, arr_error, function_number, momentum, learning_rate,\
            number_of_classes, arr_weight_bias, arr_weight_bias_output, arr_weight_bias_new, arr_weight_bias_output_new):
    arr_output_merged = []
    arr_output_merged.append(arr_Y)
    arr_output_merged.append(arr_output_nodes)
    arr_grad = []
    arr_grad.append(arr_grad_hidden)
    arr_grad.append(arr_grad_output)

    # calculate local gradient
    # iterate loop in common way but call element in reversed position
    for list_index in range(0, len(arr_grad)):
        # in case of output layer
        if(list_index == 0):
            # in case of using Sigmoid function
            if(function_number == "1"):
                for output_index in range(0, len(arr_output_nodes)):
                    if(number_of_classes == "1"):
                        arr_grad[len(arr_grad) - list_index - 1] = arr_error[output_index] * arr_output_nodes[output_index] * \
                                                                (1 - arr_output_nodes[output_index])
                    else:
                        arr_grad[len(arr_grad) - list_index - 1][output_index] = arr_error[output_index] * arr_output_nodes[output_index] * \
                                        (1 - arr_output_nodes[output_index])
            # in case of using Hyperbolic Tangent function
            elif(function_number == "2"):
                for output_index in range(0, len(arr_output_nodes)):
                    if(number_of_classes == "1"):
                        arr_grad[len(arr_grad) - list_index - 1] = arr_error[output_index] * ( 2 * arr_output_nodes[output_index] * \
                                                               (1 - arr_output_nodes[output_index]))
                    else:
                        arr_grad[len(arr_grad) - list_index - 1][output_index] = arr_error[output_index] * ( 2 * arr_output_nodes[output_index] * \
                                        (1 - arr_output_nodes[output_index]))
        #in case of hidden layers
        else:
            reversed_layer_index = len(arr_grad) - list_index - 1
            for grad_layer_index in range(0, len(arr_grad[reversed_layer_index])):
                reversed_grad_layer_index = len(arr_grad[reversed_layer_index]) - grad_layer_index - 1
                # last hidden layers -> output layer
                if(reversed_grad_layer_index == (len(arr_grad[reversed_layer_index]) - 1)):
                    if(function_number == "1"):
                        for grad_node_index in range(0, len(arr_grad[reversed_layer_index][reversed_grad_layer_index])):
                            arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += \
                            (arr_Y[reversed_grad_layer_index][grad_node_index] * (1 - arr_Y[reversed_grad_layer_index][grad_node_index]))
                            sum = 0
                            next_reversed_layer_index = reversed_layer_index + 1
                            for weight in arr_hidden_layers[len(arr_hidden_layers) - 1]:
                                if(number_of_classes == "1"):
                                    sum += weight * arr_grad[next_reversed_layer_index]
                                else:
                                    for weight_node_index in range(0, len(arr_hidden_layers[len(arr_hidden_layers) - 1])):
                                        for weight_to_node_index in range(0, len(arr_hidden_layers[len(arr_hidden_layers) - 1][weight_node_index])):
                                            sum += (arr_hidden_layers[len(arr_hidden_layers) - 1][weight_node_index][weight_to_node_index] * arr_grad[next_reversed_layer_index][grad_node_index])
                        arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += sum
                    elif(function_number == "2"):
                        for grad_node_index in range(0, len(arr_grad[reversed_layer_index][reversed_grad_layer_index])):
                            arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += \
                            (2 * arr_Y[reversed_grad_layer_index][grad_node_index] * (1 - arr_Y[reversed_grad_layer_index][grad_node_index]))
                            sum = 0
                            next_reversed_layer_index = reversed_layer_index + 1
                            if(number_of_classes == "1"):
                                for weight in arr_hidden_layers[len(arr_hidden_layers) - 1]:
                                    sum += weight * arr_grad[next_reversed_layer_index]
                            else:
                                for grad_output_index in range(0, len(arr_grad[next_reversed_layer_index])):
                                    for weight_node_index in range(0, len(arr_hidden_layers[len(arr_hidden_layers) - 1])):
                                        for weight_to_node_index in range(0, len(arr_hidden_layers[len(arr_hidden_layers) - 1][weight_node_index])):
                                            sum += (arr_hidden_layers[len(arr_hidden_layers) - 1][weight_node_index][weight_to_node_index] * arr_grad[next_reversed_layer_index][grad_output_index])
                        arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += sum
                # Input layer -> First Hidden layer 
                else:
                    if(function_number == "1"):
                        for grad_node_index in range(0, len(arr_grad[reversed_layer_index][reversed_grad_layer_index])):
                            arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += \
                            (arr_Y[reversed_grad_layer_index][grad_node_index] * (1 - arr_Y[reversed_grad_layer_index][grad_node_index]))
                            sum = 0
                            next_reversed_layer_index = reversed_layer_index + 1
                            for weight_layer_index in range(0, len(arr_hidden_layers[1])):
                                for weight_node_index in range(0, len(arr_hidden_layers[1][weight_layer_index])):
                                    for weight_to_node_index in range(0, len(arr_hidden_layers[1][weight_layer_index][weight_node_index])):
                                        sum += (arr_hidden_layers[1][weight_layer_index][weight_node_index][weight_to_node_index] * \
                                                arr_grad[reversed_layer_index][reversed_grad_layer_index + 1][grad_node_index])
                        arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += sum
                    elif(function_number == "2"):
                        for grad_node_index in range(0, len(arr_grad[reversed_layer_index][reversed_grad_layer_index])):
                            arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += \
                            (2 * arr_Y[reversed_grad_layer_index][grad_node_index] * (1 - arr_Y[reversed_grad_layer_index][grad_node_index]))
                            sum = 0
                            next_reversed_layer_index = reversed_layer_index + 1
                            for weight_layer_index in range(0, len(arr_hidden_layers[1])):
                                for weight_node_index in range(0, len(arr_hidden_layers[1][weight_layer_index])):
                                    for weight_to_node_index in range(0, len(arr_hidden_layers[1][weight_layer_index][weight_node_index])):
                                        sum += (arr_hidden_layers[1][weight_layer_index][weight_node_index][weight_to_node_index] * \
                                                arr_grad[reversed_layer_index][reversed_grad_layer_index + 1][grad_node_index])
                        arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += sum
    # calculate update weight
    for list_index in range(0, len(arr_hidden_layers)):
        # weight at the last hidden layer -> output layer
        if(list_index == 0):
            reversed_list_index = len(arr_hidden_layers) - list_index - 1
            for weight_layer_index in range(0, len(arr_hidden_layers[reversed_list_index])):
                for weight_node_index in range(0, len(arr_hidden_layers[reversed_list_index][weight_layer_index])):
                    result = 0
                    # for weight_to_node_index in range(0, len(arr_hidden_layers[reversed_list_index][weight_layer_index][weight_node_index])):
                    result += arr_hidden_layers[2][weight_layer_index][weight_node_index]
                    result += (float(momentum) * (arr_hidden_layers_new[2][weight_layer_index][weight_node_index] - arr_hidden_layers[2][weight_layer_index][weight_node_index]))
                    if(number_of_classes == "1"):
                        result += (float(learning_rate) * arr_grad[1] * arr_Y[len(arr_Y) - 1][weight_node_index])
                        result = round(result,8)
                    else:
                        for grad_node_index in range(0, len(arr_grad[1])):
                            if(weight_node_index == grad_node_index):
                                result += (float(learning_rate) * arr_grad[1][grad_node_index] * arr_Y[len(arr_Y) - 1][grad_node_index])                
                                result = round(result,8)
                    # #update weight
                    arr_hidden_layers_new[2][weight_layer_index][weight_node_index] = result
            # update weight for bias
            if(number_of_classes == "1"):
                for bias_node_index in range(0, len(arr_weight_bias_output)):
                    result = 0
                    result += (arr_weight_bias_output[bias_node_index] ) 
                    result += (float(momentum) * (arr_weight_bias_output_new[bias_node_index]  - arr_weight_bias_output[bias_node_index] ))
                    result += (float(learning_rate) * arr_grad[1] * arr_Y[len(arr_Y) - 1][weight_node_index])
                    arr_weight_bias_output_new = result
            else:
                for bias_node_index in range(0, len(arr_weight_bias_output)):
                    result = 0
                    result += (arr_weight_bias_output[bias_node_index]) 
                    result += (float(momentum) * (arr_weight_bias_output_new[bias_node_index] - arr_weight_bias_output[bias_node_index]))
                    result += (float(learning_rate) * arr_grad[1][weight_node_index] * arr_Y[len(arr_Y) - 1][weight_node_index])
                    arr_weight_bias_output_new[bias_node_index] = result
            
        # weight at an input layer -> the first hidden layer
        elif(list_index == len(arr_hidden_layers) - 1):
            reversed_list_index = len(arr_hidden_layers) - list_index - 1
            for weight_node_index in range(0, len(arr_hidden_layers[reversed_list_index])):
                for weight_to_node_index in range(0, len(arr_hidden_layers[reversed_list_index][weight_node_index])):
                    result = 0
                    result += arr_hidden_layers[0][weight_node_index][weight_to_node_index]
                    result += (float(momentum) * (arr_hidden_layers_new[0][weight_node_index][weight_to_node_index] - \
                                arr_hidden_layers[0][weight_node_index][weight_to_node_index]))
                    result += (float(learning_rate) * arr_grad[0][0][weight_node_index] * arr_input_nodes_with_value[weight_to_node_index])
                    arr_hidden_layers_new[0][weight_node_index][weight_to_node_index] = result
            # update weight bias
            for bias_node_index in range(0, len(arr_weight_bias[0])):
                result = 0
                result += arr_weight_bias[0][bias_node_index]
                result += (float(momentum) * (arr_weight_bias_new[0][bias_node_index] - \
                            arr_weight_bias[0][bias_node_index]))
                if(bias_node_index < len(arr_input_nodes_with_value)):
                    result += (float(learning_rate) * arr_grad[0][0][bias_node_index] * arr_input_nodes_with_value[bias_node_index])
                arr_weight_bias_new[0][bias_node_index] = result

        # weight at hidden layer -> hidden layer
        else:
            reversed_list_index = len(arr_hidden_layers) - list_index - 1
            for weight_layer_index in range(0, len(arr_hidden_layers[reversed_list_index])):
                for weight_node_index in range(0, len(arr_hidden_layers[reversed_list_index][weight_layer_index])):
                    for weight_to_node_index in range(0, len(arr_hidden_layers[reversed_list_index][weight_layer_index][weight_node_index])):
                        result = 0
                        result += arr_hidden_layers[reversed_list_index][weight_layer_index][weight_node_index][weight_to_node_index]
                        result += (float(momentum) * (arr_hidden_layers_new[reversed_list_index][weight_layer_index][weight_node_index][weight_to_node_index] - \
                                    arr_hidden_layers[reversed_list_index][weight_layer_index][weight_node_index][weight_to_node_index]))
                        result += (float(learning_rate) * arr_grad[0][weight_layer_index - 1][weight_node_index])
                        arr_hidden_layers_new[reversed_list_index][weight_layer_index][weight_node_index][weight_to_node_index] = result
            #update weight bias
            for bias_layer_index in range(1, len(arr_weight_bias)):
                for bias_node_index in range(0, len(arr_weight_bias[bias_layer_index])):
                    result = 0
                    result += arr_weight_bias[bias_layer_index][bias_node_index]
                    result += (float(momentum) * (arr_weight_bias_new[bias_layer_index][bias_node_index] - arr_weight_bias[bias_layer_index][bias_node_index]))
                    result += (float(learning_rate) * arr_grad[0][bias_layer_index - 1][bias_node_index])
                    arr_weight_bias[bias_layer_index][bias_node_index] = result
                                    
    #reset arr_grad
    for list_index in range(0, len(arr_grad)):
        if(list_index == 0):
            for layer_index in range(0, len(arr_grad[list_index])):
                for node_index in range(0, len(arr_grad[list_index][layer_index])):
                    arr_grad[list_index][layer_index][node_index] = 0
        else:
            arr_grad[list_index]= 0

def crossValidation(input_file, output_file, full_data_file, number_of_fold, arr_input_nodes, arr_hidden_layers, arr_hidden_layers_new, arr_hidden_layers_template, \
                    arr_Y, arr_output_nodes, arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, momentum, learning_rate, beta, arr_grad_hidden, arr_grad_output, \
                    number_of_features, number_of_layers, number_of_nodes, number_of_classes, epoch, arr_weight_bias_template, arr_weight_bias_output_template, \
                     arr_weight_bias_new, arr_weight_bias_output_new):
    data_input, dataframe_input, number_of_data_input, arr_row_input = readFile(input_file)
    data_output, dataframe_output, number_of_data_output, arr_row_output = readFile(output_file)
    data_all, dataframe_all, number_of_data_all, arr_row_all = readFile(full_data_file)
    size = math.ceil(number_of_data_input/int(number_of_fold))

    # split data into k parts
    data_chunk_input = list(chunks(arr_row_input, size))
    print("\nData chunks ...")
    print(data_chunk_input)

    # test and train
    count = 0
    all_mse = []
    all_accuracy = []
    for test_element in data_chunk_input:   
        # all_sse = []
        count_AC = 0
        count_BC = 0 
        count_AD = 0
        count_BD = 0
        count += 1
        print("------------------------------" + str(count) + " fold ------------------------------")
        test_part = test_element
        for train_element_index in range(0,len(data_chunk_input)):
            if(data_chunk_input[train_element_index] not in test_part):
                print("TRAIN----------------")
                print(data_chunk_input[train_element_index])
                print()
                print("TEST------")
                print(test_part)
                print()
                for element_index in range(0, len(data_chunk_input[train_element_index])):
                    # print("testtttt")
                    # all_sse = []
                    count_AC = 0
                    count_BC = 0 
                    count_AD = 0
                    count_BD = 0
                    for epoch_count in range(0, int(epoch)):

                        # Forwarding
                        arr_input_nodes_with_value, sse, arr_error, predicted_output, data_output_template = forward(dataframe_input, dataframe_output, data_all, data_chunk_input[train_element_index][element_index], arr_input_nodes, arr_output_nodes, arr_Y, \
                        arr_hidden_layers, arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, beta, number_of_classes)
                        
                        # Backwarding
                        arr_hidden_layers_template = copy.deepcopy(arr_hidden_layers_new)
                        arr_weight_bias_output_template = copy.deepcopy(arr_weight_bias_output_new)
                        arr_weight_bias_template = copy.deepcopy(arr_weight_bias_new)

                        backward(arr_input_nodes_with_value, arr_hidden_layers, arr_hidden_layers_new, arr_grad_hidden, arr_grad_output, arr_Y, arr_output_nodes, arr_error, function_number, \
                        momentum, learning_rate, number_of_classes, arr_weight_bias, arr_weight_bias_output, arr_weight_bias_new, arr_weight_bias_output_new)
                        arr_hidden_layers = copy.deepcopy(arr_hidden_layers_template)
                        arr_weight_bias_output = copy.deepcopy(arr_weight_bias_output_template)
                        arr_weight_bias = copy.deepcopy(arr_weight_bias_template)

                        #reset arr_Y
                        for layer_index in range(0, len(arr_Y)):
                            for node_index in range(0,len(arr_Y[layer_index])):
                                arr_Y[layer_index][node_index] = 0

                        #reset arr_output_nodes
                        for node_index in range(0, len(arr_output_nodes)):
                            arr_output_nodes[node_index] = 0
                   
                    # Testing
                    all_sse = []
                    for test_element_index in range(0, len(test_part)):
                        desired_output = []
                        print("test_part[" + str(test_element_index) + "] = " +str(test_part[test_element_index]))
                        arr_input_nodes_with_value, sse, arr_error, predicted_output, data_output_template = forward(dataframe_input, dataframe_output, data_all, test_part[test_element_index], arr_input_nodes, arr_output_nodes, arr_Y, \
                        arr_hidden_layers_new, arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, beta, number_of_classes)
                        all_sse.append(sse)
                        print("Predicted : " + str(predicted_output))
                        if(number_of_classes == "1"):
                            print("Desired Output:" + str(data_output_template[0]))
                        elif(number_of_classes == "2"):
                            desired_output.append(data_output_template[0])
                            desired_output.append(data_output_template[1])
                            print("Desired Output:" + str(desired_output))

                        if(input_file == "cross-pat-input.csv"):
                            # format output
                            if(predicted_output[0] > predicted_output[1]):
                                output = [1,0]
                            elif(predicted_output[0] < predicted_output[1]):
                                output = [0,1]
                            # check condition
                            if(output == desired_output):
                                if(desired_output == [0,1]):
                                    count_AC += 1
                                elif(desired_output == [1,0]):
                                    count_BD += 1
                            else:
                                if(desired_output == [0,1]):
                                    count_BC += 1
                                elif(desired_output == [1,0]):
                                    count_AD += 1

                    mse = calcualteMSE(all_sse, len(test_part))
                    all_sse.clear()
                    all_mse.append(mse)
                    print("MSE : " + str(mse))
                    print()
                    if(input_file == "cross-pat-input.csv"):
                        print("-------------------------------------------- CONFUSION MATRIX -----------------------------------------")
                        print("| Desire Output | -------------------------- Predicted Output -----------------------------------------")
                        print("|               |            (0,1)                                               (1,0)                 ")
                        print("|    (0,1)      |           " + str(count_AC) + "                                   " + str(count_BC) + "            ")
                        print("|    (1,0)      |           " + str(count_AD) + "                                   " + str(count_BD) + "            ")
                        print("--------------------------------------------------------------------------------------------------------")
                        accuracy = ((count_AC + count_BD)/(count_AC + count_AD + count_BC + count_BD)) * 100
                        print("                                          ACCURACY = " + str(accuracy) + " %                                  ")
                        all_accuracy.append(accuracy)

                    # #reset weight
                    arr_hidden_layers = init.createHiddenLayers(number_of_features, number_of_layers, number_of_nodes, number_of_classes) 
                    arr_hidden_layers_new = init.createHiddenLayers(number_of_features, number_of_layers, number_of_nodes, number_of_classes)
                    arr_weight_bias, arr_bias = init.createBias(number_of_nodes, number_of_layers)
                    arr_weight_bias_new, arr_bias_output_new = init.createBias(number_of_nodes, number_of_layers)
                    arr_weight_bias_output, arr_bias_output  =init.createBias(number_of_classes, 1)
                    arr_weight_bias_output_new, arr_bias_output_new  =init.createBias(number_of_classes, 1)
                    #reset arr_Y
                    for layer_index in range(0, len(arr_Y)):
                        for node_index in range(0,len(arr_Y[layer_index])):
                            arr_Y[layer_index][node_index] = 0

                    #reset arr_output_nodes
                    for node_index in range(0, len(arr_output_nodes)):
                        arr_output_nodes[node_index] = 0

                    print("------------------------------------------------------------------------------------------------------")
                    desired_output.clear()

    print("Minimum MSE : " + str(min(all_mse)))    
    print("Average MSE : " + str(sum(all_mse)/len(all_mse)))  
    if(input_file == "cross-pat-input.csv"):
        print("Average accuracy = " + str((sum(all_accuracy)/len(all_accuracy))))
