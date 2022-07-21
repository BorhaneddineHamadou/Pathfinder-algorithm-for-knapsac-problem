import numpy as np
import random
import math
import time

items = []
capacity = 0

class PFAParameters:
    def __init__(self, max_number_populations : int, max_iter_init : int, max_iter : int, bound : int):
        self.max_number_populations = max_number_populations
        self.max_iter_init = max_iter_init
        self.max_iter = max_iter
        self.bound = bound

def fitness(S):

    global items
    global capacity
    S = np.array(S)
    if (S < 0).any():
        return -1
    profit = 0
    C = capacity
    for i in range(0, len(S)):
        profit += (items[i][0]*S[i])
        C -= (items[i][1]*S[i])
        if C < 0:
            return -1
    return profit

def init_population(pfa_parameters : PFAParameters):

    global capacity
    global items

    max_iter_init = pfa_parameters.max_iter_init
    population = list()

    while (len(population) < pfa_parameters.max_number_populations) and (max_iter_init>0) :

        member = [random.randint(0, pfa_parameters.bound) for i in range(0, len(items))]
        if fitness(member) > 0:
            population.append(member)

        max_iter_init -= 1

    return population

def find_pathfinder(population):

    best_fitness = fitness(population[0])
    pathfinder = population[0]
    pathfinder_pos = 0

    for i in range(1, len(population)):
        member_fitness = fitness(population[i])
        if member_fitness > best_fitness:
            best_fitness = member_fitness
            pathfinder = population[i]
            pathfinder_pos = i

    return pathfinder,pathfinder_pos

def update_pathfinder(pathfinder_k, pathfinder_k_1, A):
    r3 = [random.random() for i in range(0, len(items))]
    pathfinder_new = np.add(pathfinder_k, np.multiply(np.multiply(2,r3), np.subtract(pathfinder_k,pathfinder_k_1)))
    pathfinder_new = np.add(pathfinder_new, A)
    return pathfinder_new

def generate_epsilon(K, Kmax, population):

    global items

    u1 = [random.uniform(-1, 1) for i in range(0, len(items))]
    ct = (1-K/Kmax)
    epsilon = np.multiply(ct, u1)
    i = random.randint(0, len(population)-1)
    j = random.randint(0, len(population)-1)
    Dij = np.abs(np.subtract(population[i], population[j]))
    epsilon = np.multiply(epsilon, Dij)
    return epsilon


def update_member(population, i, pathfinder_k, epsilon, alpha, beta):
    new_member = population[i]
    r1 = [random.random() for i in range(0, len(items))]
    R1 = np.multiply(alpha, r1)
    r2 = [random.random() for i in range(0, len(items))]
    R2 = np.multiply(beta, r2)
    j = random.randint(0, len(population)-1)

    while j == i:
        j = random.randint(0, len(population)-1)

    new_member = np.add(new_member, np.multiply(R1, np.subtract(population[i], population[j])))
    new_member = np.add(new_member, np.multiply(R2, np.subtract(pathfinder_k, population[i])))
    new_member = np.add(new_member, epsilon)
    return new_member

def generateA(Kmax, K):

    global items

    u2 = [random.uniform(-1,1) for i in range(0, len(items))]
    return np.multiply(np.exp(-2*K/Kmax),u2)

def PFA():

    global items
    global capacity

    pfa_parameters = PFAParameters(200, 30000, 30000, 6)
    population = init_population(pfa_parameters)
    pathfinder_k, pathfinder_pos = find_pathfinder(population)
    pathfinder_k_1 = pathfinder_k
    best_fitness = fitness(pathfinder_k)
    K = 1
    A = generateA(K, pfa_parameters.max_iter)
    epsilon = generate_epsilon(K,pfa_parameters.max_iter,population)

    while K < pfa_parameters.max_iter:
        print(f"{K} --> {fitness(pathfinder_k)}")
        alpha = random.uniform(1,2)
        beta = random.uniform(1,2)

        #update path finder position
        new_pathfinder = update_pathfinder(pathfinder_k, pathfinder_k_1,A)
        new_pathfinder_fitness = fitness(new_pathfinder)
        if new_pathfinder_fitness > best_fitness:
            pathfinder_k_1 = pathfinder_k
            pathfinder_k = new_pathfinder
            best_fitness = new_pathfinder_fitness


        new_population = population.copy()
        #update population positions
        for i in range(0, len(population)):
            if i == pathfinder_pos:
                continue
            new_population[i] = update_member(population, i, pathfinder_k, epsilon, alpha, beta)

        best_member, best_member_pos = find_pathfinder(new_population)
        best_member_fitness = fitness(best_member)

        if best_member_fitness > best_fitness:
            pathfinder_k_1 = pathfinder_k
            pathfinder_k = best_member
            pathfinder_pos = best_member_pos
            best_fitness = best_member_fitness

        for i in range(0, len(population)):
            if i == pathfinder_pos:
                continue
            if fitness(new_population[i]) > fitness(population[i]):
                population[i] = new_population[i]

        K += 1
        A = generateA(K, pfa_parameters.max_iter)
        epsilon = generate_epsilon(K,pfa_parameters.max_iter,population)

    return pathfinder_k


def worker():
    global items
    global capacity
    file = open("testcases.txt", 'r')
    benefices = []
    for line in file:
        values = line.split()
        if len(values) == 0:
            continue
        else:
            if values[0] == "$1":
                capacity = int(values[-1])
            elif values[0] == "$2":
                benefices = values[1:]
            elif values[0] == "$3":
                values = values[1:]
                for i in range(0, len(benefices)):
                    items.append([int(benefices[i]), int(values[i]), i])

                time_start = time.perf_counter()
                S_etoile = PFA()
                time_end = time.perf_counter()
                print("Solution S* = " + str(S_etoile))
                print(fitness(S_etoile))
                print(time_end - time_start)
worker()
