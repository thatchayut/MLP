import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def hyperbolicTangent(x):
    return math.tanh(x/2)
