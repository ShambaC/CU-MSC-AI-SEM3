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
import random

from itertools import permutations
from tqdm import tqdm, trange

def generate_population(num_cities: int, pop_size: int) -> list :
    """Method to generate the population for GA"""

    perms = permutations(range(1, num_cities+1))
    tmp_list = []
    tmp_list.extend(perms)

    population = random.sample(tmp_list, pop_size)
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

def encode(route: tuple[int], encode_list: dict) -> str :
    """Method to encode a route"""

    res = ''
    for x in route :
        res += encode_list[x]

    return res

def decode(route: str, encode_list: dict, num_cities: int) -> tuple[int] :
    """Method to decode the route"""

    encoding_size = len(route) / num_cities

    res = []
    for i in range(0, len(route), encoding_size) :
        city = encode_list[route[i : i + encoding_size]]
        res.append(city)

    return tuple(res)

def calc_fitness(route: str, num_cities: int, distance_mat: np.ndarray, encode_list: dict) -> int :
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

    return 1 / distance

def selection(routes: dict) -> np.ndarray :
    """Method to select parents based on fitness scores"""

    # Change fitness values to probabilities
    probabilities = [f / sum(routes.values()) for f in routes.values()]
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

def mutation(child: str, num_cities: int) -> str :
    """Method to mutate a child"""

    encoding_size = len(child) / num_cities
    pointA = np.random.randint(1, num_cities) * encoding_size
    pointB = np.random.randint(1, num_cities) * encoding_size

    child_res = list(child)
    child_res[pointA : pointA + encoding_size] = child[pointB : pointB + encoding_size]
    child_res[pointB : pointB + encoding_size] = child[pointA : pointA + encoding_size]
    child_res = ''.join(child_res)

    return child_res


@click.command()
@click.option('--num', '-N', default=5, help='Number of cities')
@click.option('--pop_size', '-P', default=10, help='Population size')
@click.option('--gen_count', '-G', default=50, help='Number of generations')
@click.option('--mut_prob', '-M', default=0.2, help='Mutation probability')
@click.option('--seed', '-S', default=42, help='Seed for RNG')
def main(num, pop_size, gen_count, mut_prob, seed) :

    random.seed(seed)
    np.random.seed(seed)

    # Generate distance matrx
    distance_mat = np.random.randint(5, 40, (num, num))
    # Make the matrix symmetirc
    distance_mat = ((distance_mat + distance_mat.T) / 2).astype(np.uint8)
    np.fill_diagonal(distance_mat, 0)

    # Generate population
    population = generate_population(num, pop_size)
    # Generate encoding dictionary
    encoding_dict = generate_encode_list(num)
    # Encode each route and calculate fitness
    routes_dict = {}
    for route in population :
        encoded_route = encode(route, encoding_dict)
        fitness = calc_fitness(encoded_route, num, distance_mat, encoding_dict)
        routes_dict[encoded_route] = fitness

    # Start generational loop

    for i in trange(gen_count, colour='green') :
        # Select parents
        parentA, parentB = selection(routes_dict)
        #Crossover
        childA, childB = crossover(parentA, parentB, num)
        # Mutation based on mutation probability
        if np.random.rand() < mut_prob :
            childA = mutation(childA, num)
        if np.random.rand() < mut_prob :
            childB = mutation(childB, num)

        # Calculate children fitness
        child_A_fitness = calc_fitness(childA, num, distance_mat, encoding_dict)
        child_B_fitness = calc_fitness(childB, num, distance_mat, encoding_dict)

        # Sort routes to find out routes with poorest fitness
        routes_dict = {k: v for k, v in sorted(routes_dict.items(), key=lambda item: item[1], reverse=True)}
        routes_dict.popitem()
        routes_dict.popitem()

        # Add children to population
        routes_dict[childA] = child_A_fitness
        routes_dict[childB] = child_B_fitness

    # Retrieve the route with highest fitness
    print('\n' + '#'*30 + '\n')
    routes_dict = {k: v for k, v in sorted(routes_dict.items(), key=lambda item: item[1])}
    best_route = routes_dict.popitem()
    print(f"Best route(encoded): {best_route[0]}, distance: {best_route[1]}")
    best_route = decode(best_route[0], encoding_dict, num)
    print(f"Best route(decoded): {best_route}, distance: {best_route[1]}")


if __name__ == '__main__' :
    main()