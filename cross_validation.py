import pandas
import numpy as np
import random
import math

def readFile(file):
    data = pandas.read_csv(file)
    return data

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def crossValidation(file,number_of_fold):
    data = readFile(file)
    dataframe = pandas.DataFrame(data)
    number_of_data = dataframe.shape[0] + 1
    print("Number of data : " + str(number_of_data))
    # array contains indicators of each data
    arr_row = np.arange(1,number_of_data + 1)
    # print(arr_row)
    # shuffle to decrease order dependency
    random.shuffle(arr_row)
    # print(arr_row)
    # find size of each part
    # number_of_fold = 5 # JUST FOR TEST!!!
    size = math.ceil(number_of_data/number_of_fold)
    # print(size)
    # split data into k parts
    data_chunk = list(chunks(arr_row, size))
    print("\nData chunks ...")
    print(data_chunk)

