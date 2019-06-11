import numpy as np

def replace_max(vector, num):

    '''Output the original vector to show differentiation between the original and updated vector'''
    print("Original Vector: ", vector)

    '''Sort the vector that was passed in to make finding the max value to replace easier'''
    replace_vector = np.sort(vector)
    print("Sorted Vector: ", replace_vector)

    '''Iterate through the vector so that we can replace the max value in the vector with another desired integer'''
    max_integer = replace_vector.max()
    for integer in replace_vector:
        if integer == max_integer:
            replace_vector[replace_vector == max_integer] = num

    print("Final Vector Result: ", replace_vector)


'''Create random vector with min integer being 1 and max integer being 20 with a vector size of 15'''
rand_vector = np.random.randint(low=1, high=20, size=15)
replace_max(rand_vector, 0)

