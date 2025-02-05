"""
 Quicksort in Python - Slightly adjusted 
 Source: https://www.programiz.com/dsa/quick-sort
"""


def partition(array, low, high):

    pivot = array[high][0]

    i = low - 1

    for j in range(low, high):
        if array[j][0] <= pivot:
            i = i + 1

            (array[i], array[j]) = (array[j], array[i])

    (array[i + 1], array[high]) = (array[high], array[i + 1])

    return i + 1


def quickSort(array, low, high):
    if low < high:

        pi = partition(array, low, high)

        quickSort(array, low, pi - 1)

        quickSort(array, pi + 1, high)
    return array
