import math
from random import randint
from graphics import *
import numpy as np
path = list[int]
# p1, p2 are 2 permutations
# I and J is subinterval of the interval [0, len(p1)]
# Elements from p1 with indecies in [I, J) will be copied directly to child
def partially_mapped_crossover(p1: path, p2: path, I:int, J:int):
    n = len(p1)
    
    is_placed = [False for i in range(0, n)]
    child = [-1 for i in range(0, n)]

    # Map from el to index for p1
    p1_position_of_elem= [-1 for i in range(0, n)]
    for i in range(0, n):
        p1_position_of_elem[p1[i]] = i
    
    # Map from elements of p2 to corresponding element in p1
    map_p2_to_p1 = [-1 for i in range(0, n)]
    for i in range(0, n):
        map_p2_to_p1[p2[i]] = p1[i]

    # Copy elements from selected subinterval from p1 to child
    for i in range(I, J):
        child[i] = p1[i]
        is_placed[p1[i]] = True
    
    # Elements from selected subinterval from p2 that are not present in the selected interval in p1
    conflict_elemenents = [p2[i] for i in range(I, J) if not is_placed[p2[i]]]
    for el in conflict_elemenents:
        cur_el = el
        cur_el = map_p2_to_p1[cur_el]
        while I <= p1_position_of_elem[cur_el] < J:
            cur_el = map_p2_to_p1[cur_el]
        child[p1_position_of_elem[cur_el]] = el

    # Fill empty positions of child with elements from p2
    for i in range(0, n):
        if child[i] == -1:
            child[i] = p2[i]
    
    return child

# Select random I, J and get partially mapped crossover of p1 and p2 in interval [I, J)
def partially_mapped_random_crossover(p1: path, p2:path) -> path:
    n = len(p1)
    #Get random subinterval of interval [0, n)
    a = randint(0, n-1)
    b = randint(0, n-1)
    I = min(a,b)
    J = max(a,b)
    return partially_mapped_crossover(p1, p2, I, J)

# Utility function for testing
# def sub_1(p:list[int]):
#     return [p[i] - 1 for i in range(0, len(p))]

# partially_mapped_crossover(sub_1([1,2,3,4,5,6,7]),sub_1([6,3,4,1,2,7,5]), 2, 5)
# partially_mapped_crossover(sub_1([1,5,3,4,2,7,6]),sub_1([6,3,5,1,7,4,2]), 2, 5)

# Euclidean distance between 2 points
def dist(p1:Point, p2:Point):
    return math.dist([p1.x, p1.y], [p2.x, p2.y])

# dist_matrix[i][j] = *distance between cities with indecies I and J*
def get_distance_matrix(p: list[Point]) -> list[list[float]]:
    return [[dist(p1, p2) for p1 in p] for p2 in p]

# Sum of distances between subsequent cities from path
def get_fitness(p: path, distance_matrix: list[list[float]]):
    n = len(p)
    return sum(distance_matrix[p[i-1]][p[i]] for i in range(1, n))
        # + distance_matrix[0][n-1]

def swap(p: path, i:int, j:int):
    tmp = p[i]
    p[i] = p[j]
    p[j] = tmp

def random_swap(p: path):
    n = len(p)
    i = randint(0, n-2)
    j = randint(i+1, n-1)
    swap(p, i, j)
parent_pool = list[(float, path)]

def create_pool(paths: list[path], distance_matrix: list[list[float]]) -> parent_pool:
    paths_fitness = [(get_fitness(p, distance_matrix), p) for p in paths]
    # Sort by fitness
    paths_fitness.sort(key=lambda p: p[0])
    return paths_fitness

def select_from_pool(pool: parent_pool, p: float) -> path:
    i = min(math.floor(math.fabs(np.random.normal(0, 50, 1)[0])), len(pool) - 1)
    # print(i)
    # i = min(np.random.geometric(p, 1)[0], len(pool)-1)
    return pool[i][1]

def mutate(p:path, swapC: int) -> path:
    for i in range(swapC):
        random_swap(p)
    
test_x = [
0.00
,383.46
,-27.02
,335.75
,69.43
,168.52
,320.35
,179.93
,492.67
,112.20
,306.32
,217.34]
test_y = [
0.00
,0.00
,-282.76
,-269.58
,-246.78
,31.40
,-160.90
,-318.03
,-131.56
,-110.56
,-108.09
,-447.09
]

def is_path_in_pool(p1: path, pool: parent_pool):
    return any(np.array_equal(p1, p[1]) for p in pool)
def search():
    n = 12
    gen_size = 300
    random_swap_count = 3
    base_mutation_prob = 0.2
    mutation_prob = 0.2
    mutation_prob_max = 0.7
    mutation_prob_increment = 0.05
    points = [Point(test_x[i], test_y[i]) for i in range(0, n)]
    distance_matrix = get_distance_matrix(points)
    cur_pop = [[i for i in range(0, n)] for j in range(0, gen_size)]
    strongest_paths_pool = [(9999999999999, cur_pop[0])]
    for t in range(0, 200):
        pool:parent_pool = create_pool(cur_pop, distance_matrix)

        print("Generation : ", t)
        print("BEST : ", pool[0][1])
        print("BEST_FITNESS : ", pool[0][0])
        print("MUTATION PROB: ", mutation_prob)
        print("BEST FROM GENERATION: ", )

        # If no better path has been found - increase mutation probability
        if pool[0][0] < strongest_paths_pool[0][0]:
            # random_swap_count = initial_random_swap_count
            mutation_prob = base_mutation_prob
        else:
            # random_swap_count = int(min(random_swap_count + 1, n/2))
            mutation_prob = min(mutation_prob_max, mutation_prob + mutation_prob_increment)
            # 
            if mutation_prob == mutation_prob_max:
                strongest_paths_pool = [strongest_paths_pool[0]]

        # Keep track of best paths in each generation
        if not is_path_in_pool(pool[0][1], strongest_paths_pool):
            strongest_paths_pool.append(pool[0])
            strongest_paths_pool.sort(key=lambda p: p[0])
            

        new_pop = []
        for cI in range(0, gen_size):
            child = None
            P = randint(0, 100)
            # Select child from previous strongest paths
            if P < 10:
                child = select_from_pool(strongest_paths_pool, 0.4).copy()
            # Create child as a result of crossover of 2 paths
            elif P <= 90:
                p1 = select_from_pool(pool, 0.01)
                p2 = select_from_pool(pool, 0.01)
                child = partially_mapped_random_crossover(p1, p2)
            # Path from prev generation will be path in new generation
            else:
                child = select_from_pool(pool, 0.01).copy()
            # Small chance for mutation of each child
            if randint(0, 100)/100. < mutation_prob:
                mutate(child, random_swap_count)
            new_pop.append(child)

        cur_pop = new_pop

    # print(strongest_paths_pool)
    return strongest_paths_pool[0]

search()

# # input()
# K = 0
# N = 100
# for i in range(0, N):
#     res = search()
#     if(res[0] - 1595.738522033024 < 0.01):
#         K+=1
#     print(K,i,N)