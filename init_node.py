import numpy as np

def createInputNodes(number_of_features):
    arr_input_nodes = np.zeros(int(number_of_features))
    return arr_input_nodes

def createOutputNodes(number_of_classes):
    arr_output_nodes = np.zeros(int(number_of_classes))
    return arr_output_nodes

def createHiddenLayers(number_of_features,number_of_layers,number_of_nodes,number_of_classes):
    # first hidden layer that is connected with input nodes
    first_layer = []
    last_layer = []
    node = []
    layer = []
    final = []
    count = 0
    np.random.seed(1)
    while count < int(number_of_nodes):
        arr = np.random.uniform(low=-1.0,high=1.0,size=int(number_of_features))
        first_layer.append(arr)
        count += 1
    final.append(first_layer)
    # the rest hidden layers
    # create all hidden layers except the first and the last layer that
    # are connected to input nodes and output nodes respectively
    
    for layer_count in range(0, int(number_of_layers)):
        for node_count in range(0,int(number_of_nodes)):
            arr = np.random.uniform(low=-1.0,high=1.0,size=int(number_of_nodes))
            node.append(arr)
            new_arr = np.array(node)
        layer.append(new_arr)
        node.clear()
    final.append(layer)

    #The last hidden layer connected to an output layer
    count = 0
    while count < int(number_of_nodes):
        arr = np.random.uniform(low=-1.0,high=1.0,size=int(number_of_classes))
        last_layer.append(arr)
        count += 1
    final.append(last_layer)
    return final

def createBias(number_of_nodes,number_of_layers):
    weight_bias = []
    node1 = []
    layer1 = []
    node2 = []
    layer2 =[]
    bias = []
    np.random.seed(1)
    #initial weight for each bias
    for layer_count in range(0,int(number_of_layers)):
        arr = np.random.uniform(low=-1.0,high=1.0,size=int(number_of_nodes))
        weight_bias.append(arr)

    #initial bias as 1
    for layer_count in range(0,int(number_of_layers)):
        arr = np.ones(int(number_of_nodes))
        bias.append(arr)
    return weight_bias, bias


def createY(number_of_nodes, number_of_layers):
    node = []
    layer = []
    # final = []
    for layer_count in range(0, int(number_of_layers)):
        arr = np.zeros(int(number_of_nodes))
        layer.append(arr)
    return layer 

def createLocalGradOutput(number_of_classes):
    arr_grad_output = np.zeros(int(number_of_classes))
    return arr_grad_output

def createLocalGradHidden(number_of_nodes, number_of_layers):
    node = []
    arr_grad_hidden = []
    for layer_count in range(0, int(number_of_layers)):
        arr = np.zeros(int(number_of_nodes))
        arr_grad_hidden.append(arr)
    return arr_grad_hidden 
