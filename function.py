import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def hyperbolicTangent(x):
    return math.tanh(x/2)

def unitStep(x,beta):
    if (x >= beta):
        y = 1
        return y
    elif (x < beta):
        y = -1
        return y

def ramp(x,beta):
    if(x >= beta):
        y = 1
        return y
    elif(-beta < x < beta):
        y = x
        return y
    elif(x <= -beta):
        y = -1
        return y