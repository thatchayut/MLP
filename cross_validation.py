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
    # print("merged_data" + str(merged_data))
    min_value = min(merged_data)
    max_value = max(merged_data)
    # print("MIN value : " + str(min_value))
    # print("MAX value : " + str(max_value))
    for element in input_data:
        # data[index] = (data[index] - mean_value)/(max_value - min_value)
        result = (element - min_value)/(max_value - min_value)
        result = round(result,5)
        normalized_input_data.append(result)
        # print("testtttttttttt")
    # print("normalized_input_data : " + str(normalized_input_data))
    normalized_output_data = []
    for element in output_data:
        # data[index] = (data[index] - mean_value)/(max_value - min_value)
        result = (element - min_value)/(max_value - min_value)
        result = round(result,5)
        normalized_output_data.append(result)
        # print("testtttttttttt")
    # print("normalized_output_data : " + str(normalized_output_data))

    return normalized_input_data, normalized_output_data

def convertBack(value, input_data, output_data):
    merged_data = [] 
    merged_data.extend(input_data)
    merged_data.extend(output_data)
    # print("merged_data" + str(merged_data))
    min_value = min(merged_data)
    max_value = max(merged_data)

    new_value = (value * (max_value - min_value)) + min_value
    return new_value

def normalizeError(value, input_data, output_data):
    merged_data = [] 
    merged_data.extend(input_data)
    merged_data.extend(output_data)
    # print("merged_data" + str(merged_data))
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
        # error_percentage = ((error_value/actual_output[index]) * 100)
        arr_error.append(error_value)
        # print("absolute error of output #" + str(index) + " = " + str(error_value) + "(" + str(error_percentage) + " %)")
    # print("arr _error : " + str(arr_error))
    # calculate Sum Square Error (SSE)
    for element in arr_error:
        sse += (1/2)*(element * element)
    # print("sse : " + str(sse))
    return sse, arr_error

def calcualteMSE(arr_error, size):
    result = 0
    for element in arr_error:
        result += element
    result = result/size
    return result

def forward (dataframe_input, dataframe_output, data_all, line, arr_input_nodes, arr_output_nodes, arr_Y, arr_hidden_layers,\
            arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, beta, number_of_classes):
    # calculate min, max, mean to be used in feature scaling
    # min_value = data_all.min()
    # max_value = data_all.max()
    # mean_value = data_all.mean()
    # mean_value = round(mean_value, 5)
    # print("MIN from ALL : " + str(min_value[1]))
    # print("MAX from ALL : " + str(max_value[1]))
    # print("MEAN from ALL : " + str(mean_value[1]))
    # change number of line in to dataframe
    line = line - 2
    # print("line : " + str(line + 2))
    data_input = dataframe_input.iloc[line]
    data_input_template = copy.deepcopy(data_input)
    # print(data_input)
    # data_input = featureScaling(data_input)
    # print(data_input)
    data_output = dataframe_output.iloc[line]
    data_output_template = copy.deepcopy(data_output)
    # print(data_output)
    # data_output = featureScaling(data_output)
    data_input, data_output = featureScaling(data_input, data_output)
    # print(len(data_input))

    # check if input nodes are enough
    input_check = False
    if (len(data_input) == len(arr_input_nodes)):
        # print("Input later : OK")
        # print()
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
        # print("input : " + str(arr_input_nodes))
        # print()
    
    # check if output nodes are enough
    output_check = False
    if (len(data_output) == len(arr_output_nodes)):
        # print("Output layer : OK")
        # print()
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
        # print("BEFORE...")
        # print("arr_output_nodes : " + str(arr_output_nodes))
        # print("arr_Y : " + str(arr_Y))
        # print()
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
                        # print("BEFORE -> arr_output_nodes[" +  str(output_index) + "] = " + str(arr_output_nodes[output_index]))
                        arr_output_nodes[output_index] = useFunction(arr_output_nodes[output_index], function_number, beta)
                        # print("AFTER -> arr_output_nodes[" +  str(output_index) + "] = " + str(arr_output_nodes[output_index]))
                else:
                    for output_index in range(0, len(arr_output_nodes)):
                        for weight_node_index in range(0, len(arr_hidden_layers[2])):
                            result = 0
                            for weight_to_node_index in range(0, len(arr_hidden_layers[2][weight_node_index])):
                                result += (arr_hidden_layers[2][weight_node_index][weight_to_node_index] * arr_Y[len(arr_Y) - 1][weight_node_index])
                                result += (arr_weight_bias_output[output_index] * arr_bias_output[output_index])
                            arr_output_nodes[output_index] = result
                            # print("arr_output_nodes[" +  str(output_index) + "] = " + str(arr_output_nodes[output_index]))
                            arr_output_nodes[output_index] = useFunction(arr_output_nodes[output_index], function_number, beta)
                            # print("AFTER -> arr_output_nodes[" +  str(output_index) + "] = " + str(arr_output_nodes[output_index]))
            # y at the first hidden layer
            elif(layer_index == 0):
                # for arr_Y_node_index in range(0, len(arr_Y[0])):
                for weight_node_index in range(0, len(arr_hidden_layers[0])):
                    result = 0
                    for weight_to_node_index in range(0, len(arr_hidden_layers[0][weight_node_index])):
                        result += (arr_input_nodes[weight_node_index] * arr_hidden_layers[0][weight_node_index][weight_to_node_index])
                    result += (arr_bias[0][weight_node_index] * arr_weight_bias[0][weight_node_index])
                    arr_Y[0][weight_node_index] = result
                    # print("BEFORE -> arr_Y[0][" + str(weight_node_index) + "] = " + str(arr_Y[0][weight_node_index]))
                    arr_Y[0][weight_node_index] = useFunction(arr_Y[0][weight_node_index], function_number, beta)
                    # print("AFTER -> arr_Y[0][" + str(weight_node_index) + "] = " + str(arr_Y[0][weight_node_index]))
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
                                # print("BEFORE -> arr_Y{" + str(arr_Y_layer_index) + "][" + str(arr_Y_node_index) + "] = " + str(arr_Y[arr_Y_layer_index][arr_Y_node_index]))
                                arr_Y[arr_Y_layer_index][arr_Y_node_index] = useFunction(arr_Y[arr_Y_layer_index][arr_Y_node_index], function_number, beta)
                                # print("AFTER -> arr_Y{" + str(arr_Y_layer_index) + "][" + str(arr_Y_node_index) + "] = " + str(arr_Y[arr_Y_layer_index][arr_Y_node_index]))
        # print()
        # print("AFTER...")
        # print("arr_Y" + str(arr_Y))
        # print("arr_output_nodes(actual output) : " + str(arr_output_nodes))
        # print("data output(desired output)  : " + str(data_output))
        converted_arr_output_node = []
        for element_index in range(0, len(arr_output_nodes)):
            converted_value = convertBack(arr_output_nodes[element_index], data_input_template, data_output_template)
            converted_arr_output_node.append(converted_value)
        # print("actual output : " + str(converted_arr_output_node))
        # print("desired output : " + str(data_output_template) )

        sse, arr_error = calculateError(converted_arr_output_node, data_output_template)
        # print("SSE = " + str(sse))
        # print("sse = " + str(sse))
        predicted_output = copy.deepcopy(converted_arr_output_node)
        converted_arr_output_node.clear()
        
        #normalize error
        normalized_arr_error = []
        for element_index in range(0, len(arr_error)):
            error = normalizeError(arr_error[element_index], data_input_template, data_output_template)
            normalized_arr_error.append(error)

        # sse, arr_error = calculateError(arr_output_nodes, data_output)
        # return arr_input_nodes, sse, arr_error
        return arr_input_nodes, sse, normalized_arr_error, predicted_output, data_output_template
    else:
        print("cannot do FORWARDING!")
        print()

def backward(arr_input_nodes_with_value, arr_hidden_layers, arr_hidden_layers_new, arr_grad_hidden, arr_grad_output, arr_Y, arr_output_nodes, arr_error, function_number, momentum, learning_rate,\
            number_of_classes):
    arr_output_merged = []
    arr_output_merged.append(arr_Y)
    arr_output_merged.append(arr_output_nodes)
    arr_grad = []
    arr_grad.append(arr_grad_hidden)
    arr_grad.append(arr_grad_output)
    # print("INPUT : " + str(arr_input_nodes_with_value))
    # print("BEFORE.......")
    # print("arr_Y : " + str(arr_Y))
    # print("arr_output_merged" + str(arr_output_merged))
    # print("arr_grad_hidden, arr_grad_output" + str(arr_grad))
    # print("arr_error : " + str(arr_error))
    # calculate local gradient
    # iterate loop in common way but call element in reversed position
    for list_index in range(0, len(arr_grad)):
        # in case of output layer
        if(list_index == 0):
            # in case of using Sigmoid function
            if(function_number == "1"):
                for output_index in range(0, len(arr_output_nodes)):
                    arr_grad[len(arr_grad) - list_index - 1] = arr_error[output_index] * arr_output_nodes[output_index] * \
                                                               (1 - arr_output_nodes[output_index])
            # in case of using Hyperbolic Tangent function
            elif(function_number == "2"):
                for output_index in range(0, len(arr_output_nodes)):
                    arr_grad[len(arr_grad) - list_index - 1] = arr_error[output_index] * ( 2 * arr_output_nodes[output_index] * \
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
                            # for grad_output_node in range(0, len(arr_grad[next_reversed_layer_index])):
                            for weight in arr_hidden_layers[len(arr_hidden_layers) - 1]:
                                if(number_of_classes == "1"):
                                    sum += weight * arr_grad[next_reversed_layer_index]
                                else:
                                    sum += weight * arr_grad[next_reversed_layer_index][grad_node_index]
                        arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += sum
                    elif(function_number == "2"):
                        for grad_node_index in range(0, len(arr_grad[reversed_layer_index][reversed_grad_layer_index])):
                            arr_grad[reversed_layer_index][reversed_grad_layer_index][grad_node_index] += \
                            (2 * arr_Y[reversed_grad_layer_index][grad_node_index] * (1 - arr_Y[reversed_grad_layer_index][grad_node_index]))
                            sum = 0
                            next_reversed_layer_index = reversed_layer_index + 1
                            # for grad_output_node in range(0, len(arr_grad[next_reversed_layer_index])):
                            for weight in arr_hidden_layers[len(arr_hidden_layers) - 1]:
                                if(number_of_classes == "1"):
                                    sum += weight * arr_grad[next_reversed_layer_index]
                                else:
                                    sum += weight * arr_grad[next_reversed_layer_index][grad_node_index]
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
                                        # print("arr_hidden_layers[1][" + str(weight_layer_index) + "][" + str(weight_node_index) + "][" + str(weight_to_node_index) + "]")
                                        # print("arr_grad[" + str(reversed_layer_index) + "][" + str(reversed_grad_layer_index + 1) + "][" + str(grad_node_index) + "]")
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
                                        # print("arr_hidden_layers[1][" + str(weight_layer_index) + "][" + str(weight_node_index) + "][" + str(weight_to_node_index) + "]")
                                        # print("arr_grad[" + str(reversed_layer_index) + "][" + str(reversed_grad_layer_index + 1) + "][" + str(grad_node_index) + "]")
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
                    # print("BEFORE UPDATE -> arr_hidden_layers_new[2]["+str(weight_layer_index) + "][" + str(weight_node_index) + "]" \
                        # + " = " + str(arr_hidden_layers_new[2][weight_layer_index][weight_node_index]) )
                    result += arr_hidden_layers[2][weight_layer_index][weight_node_index]
                    result += (float(momentum) * (arr_hidden_layers_new[2][weight_layer_index][weight_node_index] - arr_hidden_layers[2][weight_layer_index][weight_node_index]))
                    if(number_of_classes == "1"):
                        result += (float(learning_rate) * arr_grad[1] * arr_Y[len(arr_Y) - 1][weight_node_index])
                        result = round(result,8)
                    else:
                        result += (float(learning_rate) * arr_grad[1][weight_node_index] * arr_Y[len(arr_Y) - 1][weight_node_index])                
                    # #update weight
                    arr_hidden_layers_new[2][weight_layer_index][weight_node_index] = result
                    # print("AFTER UPDATE -> arr_hidden_layers_new[2]["+str(weight_layer_index) + "][" + str(weight_node_index) + "]" \
                            # + " = " + str(arr_hidden_layers_new[2][weight_layer_index][weight_node_index]))
        # weight at an input layer -> the first hidden layer
        elif(list_index == len(arr_hidden_layers) - 1):
            reversed_list_index = len(arr_hidden_layers) - list_index - 1
            for weight_node_index in range(0, len(arr_hidden_layers[reversed_list_index])):
                for weight_to_node_index in range(0, len(arr_hidden_layers[reversed_list_index][weight_layer_index])):
                    result = 0
                    # for weight_to_node_index in range(0, len(arr_hidden_layers[reversed_layer_index][weight_layer_index][weight_node_index])):
                    # print("BEFORE UPDATE -> arr_hidden_layers_new[0]["+str(weight_node_index) + "][" + str(weight_to_node_index) + \
                    # "] = " + str(arr_hidden_layers_new[0][weight_node_index][weight_to_node_index]) )
                    result += arr_hidden_layers[0][weight_node_index][weight_to_node_index]
                    result += (float(momentum) * (arr_hidden_layers_new[0][weight_node_index][weight_to_node_index] - \
                                arr_hidden_layers[0][weight_node_index][weight_to_node_index]))
                    result += (float(learning_rate) * arr_grad[0][0][weight_node_index] * arr_input_nodes_with_value[weight_node_index])
                    arr_hidden_layers_new[0][weight_node_index][weight_to_node_index] = result
                    # print("AFTER UPDATE -> arr_hidden_layers_new[0]["+str(weight_node_index) + "][" + str(weight_to_node_index) + \
                    # "] = " + str(arr_hidden_layers_new[0][weight_node_index][weight_to_node_index]))
        # weight at hidden layer -> hidden layer
        else:
            reversed_list_index = len(arr_hidden_layers) - list_index - 1
            for weight_layer_index in range(0, len(arr_hidden_layers[reversed_list_index])):
                for weight_node_index in range(0, len(arr_hidden_layers[reversed_list_index][weight_layer_index])):
                    for weight_to_node_index in range(0, len(arr_hidden_layers[reversed_list_index][weight_layer_index][weight_node_index])):
                        result = 0
                        # print("BEFORE UPDATE -> arr_hidden_layers_new[1]["+str(weight_layer_index) + "][" + str(weight_node_index) \
                        # + "][" + str(weight_to_node_index) +"] = " + str(arr_hidden_layers_new[1][weight_layer_index][weight_node_index][weight_to_node_index]) )
                        result += arr_hidden_layers[reversed_list_index][weight_layer_index][weight_node_index][weight_to_node_index]
                        result += (float(momentum) * (arr_hidden_layers_new[reversed_list_index][weight_layer_index][weight_node_index][weight_to_node_index] - \
                                    arr_hidden_layers[reversed_list_index][weight_layer_index][weight_node_index][weight_to_node_index]))
                        result += (float(learning_rate) * arr_grad[0][weight_layer_index - 1][weight_node_index])
                        arr_hidden_layers_new[reversed_list_index][weight_layer_index][weight_node_index][weight_to_node_index] = result
                        # print("AFTER UPDATE -> arr_hidden_layers_new[1]["+str(weight_layer_index) + "][" + str(weight_node_index) \
                        # + "][" + str(weight_to_node_index) +"] = " + str(arr_hidden_layers_new[1][weight_layer_index][weight_node_index][weight_to_node_index]) )
    # print("AFTER.......")
    # print("arr_Y : " + str(arr_Y))
    # print("arr_output_merged" + str(arr_output_merged))
    # print("arr_grad_hidden, arr_grad_output" + str(arr_grad))
    # print("arr_error : " + str(arr_error))

    #reset arr_grad
    for list_index in range(0, len(arr_grad)):
        if(list_index == 0):
            for layer_index in range(0, len(arr_grad[list_index])):
                for node_index in range(0, len(arr_grad[list_index][layer_index])):
                    arr_grad[list_index][layer_index][node_index] = 0
        else:
            # for output_grad_index in range(0, len(arr_grad[list_index])):
            arr_grad[list_index]= 0

def crossValidation(input_file, output_file, full_data_file, number_of_fold, arr_input_nodes, arr_hidden_layers, arr_hidden_layers_new, arr_hidden_layers_template, \
                    arr_Y, arr_output_nodes, arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, momentum, learning_rate, beta, arr_grad_hidden, arr_grad_output, \
                    number_of_features, number_of_layers, number_of_nodes, number_of_classes, epoch):
    data_input, dataframe_input, number_of_data_input, arr_row_input = readFile(input_file)
    data_output, dataframe_output, number_of_data_output, arr_row_output = readFile(output_file)
    data_all, dataframe_all, number_of_data_all, arr_row_all = readFile(full_data_file)
    # print(dataframe_output)
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
        all_mse = []
        # all_sse = []
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
                    # all_sse = []
                    for epoch_count in range(0, int(epoch)):
                        # print("*****************************************************************************************************")
                        # print("                                           FORWARD                                                   ")
                        # print("*****************************************************************************************************")
                        arr_input_nodes_with_value, sse, arr_error, predicted_output, data_output_template = forward(dataframe_input, dataframe_output, data_all, data_chunk_input[train_element_index][element_index], arr_input_nodes, arr_output_nodes, arr_Y, \
                        arr_hidden_layers, arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, beta, number_of_classes)
                        # all_sse.append(sse)

                        # print("*****************************************************************************************************")
                        # print("                                           BACKWARD                                                   ")
                        # print("*****************************************************************************************************")
                        arr_hidden_layers_template = copy.deepcopy(arr_hidden_layers_new)
                        # print("arr_hidden_layers_template = ")
                        # print("arr_hidden_layers = ")
                        # print(str(arr_hidden_layers))
                        # print(str(arr_hidden_layers_template))
                        backward(arr_input_nodes_with_value, arr_hidden_layers, arr_hidden_layers_new, arr_grad_hidden, arr_grad_output, arr_Y, arr_output_nodes, arr_error, function_number, \
                        momentum, learning_rate, number_of_classes)
                        arr_hidden_layers = copy.deepcopy(arr_hidden_layers_template)
                        # print("arr_hidden_layers = ")
                        # print(str(arr_hidden_layers))

                        #reset arr_Y
                        for layer_index in range(0, len(arr_Y)):
                            for node_index in range(0,len(arr_Y[layer_index])):
                                arr_Y[layer_index][node_index] = 0
                        # print("arr_Y after reset: " + str(arr_Y))

                        #reset arr_output_nodes
                        for node_index in range(0, len(arr_output_nodes)):
                            arr_output_nodes[node_index] = 0
                    
                    # #testing
                    # for test_element_index in range(0, len(test_part)):
                    # if(element_index == test_element_index):
                    # print("*****************************************************************************************************")
                    # print("                                           TESTING                                                   ")
                    # print("*****************************************************************************************************")
                    # print("arr_hidden_layers : " + str(arr_hidden_layers))
                    # print("arr_hidden_layers_new : " + str(arr_hidden_layers_new))
                    # print("arr_output_nodes : " + str(arr_output_nodes))
                    # print("arr_Y : " + str(arr_Y))
                    # all_sse = []
                    all_sse = []
                    for test_element_index in range(0, len(test_part)):
                    # if (element_index < len(test_part)):
                        print("test_part[" + str(element_index) + "] = " +str(test_part[test_element_index]))
                        arr_input_nodes_with_value, sse, arr_error, predicted_output, data_output_template = forward(dataframe_input, dataframe_output, data_all, test_part[test_element_index], arr_input_nodes, arr_output_nodes, arr_Y, \
                        arr_hidden_layers_new, arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, beta, number_of_classes)
                        all_sse.append(sse)
                        print("Predicted : " + str(predicted_output))
                        print("Desired Output : " + str(data_output_template[0]))
                    print("all_sse : " + str(all_sse))
                    print("number of all sse : " + str(len(all_sse)))
                    mse = calcualteMSE(all_sse, len(test_part))
                    all_mse.append(mse)
                    print("MSE : " + str(mse))
                    # print("MSE (" + str(element_index) + ") : " + str(mse))

                    # #reset weight
                    arr_hidden_layers = init.createHiddenLayers(number_of_features, number_of_layers, number_of_nodes, number_of_classes) 
                    arr_hidden_layers_new = init.createHiddenLayers(number_of_features, number_of_layers, number_of_nodes, number_of_classes)
                
                    #reset arr_Y
                    for layer_index in range(0, len(arr_Y)):
                        for node_index in range(0,len(arr_Y[layer_index])):
                            arr_Y[layer_index][node_index] = 0
                    # print("arr_Y after reset: " + str(arr_Y))

                    #reset arr_output_nodes
                    for node_index in range(0, len(arr_output_nodes)):
                        arr_output_nodes[node_index] = 0

                    # print("arr_output_nodes after reset : " + str(arr_output_nodes))
                    print("------------------------------------------------------------------------------------------------------")
        # print("Number of test data : " + str(len(test_part)))
        # print("all_sse : " + str(len(all_sse)))
        # mse = calcualteMSE(all_sse, len(test_part))
        # print("MSE = " + str(mse))
        print("Minimum MSE : " + str(min(all_mse)))      
        print()
                # mse = calcualteMSE(all_sse, number_of_data_all)
                # print("MSE : " + str(mse))
                # print("arr_hidden_layers : ")
                # print(arr_hidden_layers)
                # print("arr_hidden_layers_new : ")
                # print(arr_hidden_layers_new)
                # print("arr_hidden_layers_template : ")
                # print(arr_hidden_layers_template)
                # print()
            # for element in test_part:
            #         print("*****************************************************************************************************")
            #         print("                                           TESTING                                                   ")
            #         print("*****************************************************************************************************")
            #         all_sse = []
            #         arr_input_nodes_with_value, sse, arr_error = forward(dataframe_input, dataframe_output, data_all, element, arr_input_nodes, arr_output_nodes, arr_Y, \
            #         arr_hidden_layers, arr_weight_bias, arr_bias, arr_weight_bias_output, arr_bias_output, function_number, beta)
            #         all_sse.append(sse)

        # print("TEST------")
        # print(test_part)
        # print()