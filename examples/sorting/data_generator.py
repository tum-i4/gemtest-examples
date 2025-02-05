import random

def generate_data(n=20, min_value=1, max_value=100):
    """
    Data generation function, returns a list of tuples with random integers and unique identifiers.
    """
    data = []
    for i in range(n):
        value = random.randint(min_value, max_value)
        data.append(value)
    return data

def generate_data_with_identifiers(n=20, min_value=1, max_value=100):
    """
    Data generation function, returns a list of tuples with random integers and unique identifiers.
    """
    data = []
    for i in range(n):
        value = random.randint(min_value, max_value)
        identifier = f"id_{i}"
        data.append((value, identifier))
    return data