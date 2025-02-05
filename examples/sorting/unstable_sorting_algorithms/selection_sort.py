"""
 Selection Sort in Python - Slightly adjusted 
 Source: https://www.programiz.com/dsa/selection-sort
"""

def selectionSort(array, size):
    for step in range(size):
        min_idx = step

        for i in range(step + 1, size):
            if array[i][0] < array[min_idx][0]:
                min_idx = i

        (array[step], array[min_idx]) = (array[min_idx], array[step])
    return array
