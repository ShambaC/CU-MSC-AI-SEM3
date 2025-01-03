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

def encode(route: tuple[int], encode_list: dict) -> tuple[str] :
    """Method to encode a route"""

    res = []
    for x in route :
        res.append(encode_list[x])

    return tuple(res)

def decode(route: tuple[str], encode_list: dict) -> tuple[int] :
    """Method to decode the route"""

    res = []
    for x in route :
        city = encode_list[x]
        res.append(city)

    return tuple(res)

def calc_fitness(route: tuple[str], distance_mat: np.ndarray, encode_list: dict) -> int :
    """Calculate total distance of a route"""

    distance = 0
    
    for i in range(len(route)-1) :
        city_A = encode_list[route[i]] - 1
        city_B = encode_list[route[i+1]] - 1

        distance += distance_mat[city_A][city_B]

    # Add distance for return trip
    city_A = encode_list[route[0]] - 1
    city_B = encode_list[route[-1]] - 1
    distance += distance_mat[city_A][city_B]

    return 1 / distance

def selection(routes: dict) -> np.ndarray :
    """Method to select parents based on fitness scores"""

    # Change fitness values to probabilities
    probabilities = [f / sum(routes.values()) for f in routes.values()]
    parents_idx = np.random.choice(len(routes.keys()), 2, p=probabilities)
    parents = [list(routes.keys())[i] for i in parents_idx]

    return parents

def crossover(parentA: tuple[str], parentB: tuple[str], num_cities: int) -> tuple[tuple[str]] :
    """Method to crossover two parents to produce two children"""

    # Single point crossover
    point = np.random.randint(1, num_cities)

    childA = parentA[:point] + parentB[point:]
    childB = parentB[:point] + parentA[point:]

    return (childA, childB)

def mutation(child: tuple[str], num_cities: int) -> tuple[str] :
    """Method to mutate a child"""

    pointA = np.random.randint(1, num_cities)
    pointB = np.random.randint(1, num_cities)

    child_res = list(child)
    child_res[pointA], child_res[pointB] = child[pointB], child[pointA]

    return tuple(child_res)


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
        fitness = calc_fitness(encoded_route, distance_mat, encoding_dict)
        routes_dict[encoded_route] = fitness

    # Start generational loop
    # Flag for low population check
    flag = False

    for i in trange(gen_count, colour='green') :
        # Check for low population
        if len(routes_dict) == 1 :
            flag = True
            break

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
        child_A_fitness = calc_fitness(childA, distance_mat, encoding_dict)
        child_B_fitness = calc_fitness(childB, distance_mat, encoding_dict)

        # Sort routes to find out routes with poorest fitness
        routes_dict = {k: v for k, v in sorted(routes_dict.items(), key=lambda item: item[1], reverse=True)}
        routes_dict.popitem()
        routes_dict.popitem()

        # Add children to population
        routes_dict[childA] = child_A_fitness
        routes_dict[childB] = child_B_fitness

    # Retrieve the route with highest fitness
    if flag :
        print("Population has become too low, cannot procreate anymore. ABORTING!")

    print('\n' + '#'*30 + '\n')
    routes_dict = {k: v for k, v in sorted(routes_dict.items(), key=lambda item: item[1])}
    best_route = routes_dict.popitem()
    distance = best_route[1]
    print(f"Best route(encoded): {''.join(best_route[0])}, fitness: {distance}")
    best_route = decode(best_route[0], encoding_dict)
    print(f"Best route(decoded): {best_route}, fitness: {distance}")


if __name__ == '__main__' :
    main()