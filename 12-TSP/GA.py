"""GA for Travelling Salesman Problem"""

"""
Method: 
- Population creation
- Encoding
- Selection based on weighted probability, parents with good fitness score
- Crossover
- Mutation

No elitism
"""

import numpy as np
import click

from itertools import permutations
from random import sample
from tqdm import tqdm

def generate_population(num_cities: int, pop_size: int) -> list :
    """Method to generate the population for GA"""

    perms = permutations(range(1, num_cities+1))
    tmp_list = []
    tmp_list.extend(perms)

    population = sample(tmp_list, pop_size)
    return population

def generate_encode_list(num_cities: int) -> dict :
    """Method to generate encodings for each city"""

    pad = len(bin(num_cities)[2:])
    res = {}
    for i in range(1, num_cities+1) :
        res[i] = bin(i)[2:].zfill(pad)

    rev_dict = {v: k for k, v in res.items()}
    res.update(rev_dict)

    return res

def encode(route: tuple, encode_list: dict) -> str :
    """Method to encode a route"""

    res = ''
    for x in route :
        res += encode_list[x]

    return res

def calc_distance(route: str, num_cities: int, distance_mat: np.ndarray, encode_list: dict) -> int :
    """Calculate total distance of a route"""

    distance = 0
    encode_size = len(route) / num_cities
    
    for i in range(0, len(route)-encode_size, encode_size) :
        city_A = encode_list[route[i:i+encode_size]]
        city_B = encode_list[route[i+encode_size:i+(2*encode_size)]]

        distance += distance_mat[city_A][city_B]

    # Add distance for return trip
    city_A = encode_list[route[0:encode_size]]
    city_B = encode_list[route[-encode_size:]]
    distance += distance_mat[city_A][city_B]

    return distance

def selection(routes: dict) -> np.ndarray :
    """Method to select parents based on fitness scores"""

    probabilities = [d / sum(routes.values()) for d in routes.values()]
    parents = np.random.choice(routes.keys(), 2, p=probabilities)

    return parents

def crossover(parentA: str, parentB: str, num_cities: int) -> tuple[str] :
    """Method to crossover two parents to produce two children"""

    # Single point crossover
    encoding_size = len(parentA) / num_cities
    point = np.random.randint(1, num_cities) * encoding_size

    childA = parentA[:point] + parentB[point:]
    childB = parentB[:point] + parentA[point:]

    return (childA, childB)

def mutation(child: str, num_cities: int) -> tuple[str] :
    """Method to mutate a child"""

    encoding_size = len(child) / num_cities
    pointA = np.random.randint(1, num_cities) * encoding_size
    pointB = np.random.randint(1, num_cities) * encoding_size

    child_res = list(child)
    child_res[pointA : pointA + encoding_size], child_res[pointB : pointB + encoding_size] = child[pointB : pointB + encoding_size], child[pointA : pointA + encoding_size]
    child_res = ''.join(child_res)

    return child_res


@click.command()
@click.option('--num', '-N', default=5, help='Number of cities')
@click.option('--pop_size', '-P', default=10, help='Population size')
@click.option('--gen_count', '-G', default=50, help='Number of generations')
@click.option('--mut_rate', '-M', default=0.2, help='Mutation rate')
def main(num, pop_size, gen_count, mut_rate) :

    # Generate distance matrx
    distance_mat = np.random.randint(5, 40, (num, num))
    # Make the matrix symmetirc
    distance_mat = ((distance_mat + distance_mat.T) / 2).astype(np.uint8)
    np.fill_diagonal(distance_mat, 0)