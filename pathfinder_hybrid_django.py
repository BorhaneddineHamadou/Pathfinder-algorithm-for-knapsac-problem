import numpy as np
import random
import math
import time

items = []
capacity = 0
umax = None
rmax = None


class PFAParameters:
    # parametres de l'agorithme PFA
    def __init__(self, taille_initial_population: int, max_iter_init: int, max_iter: int, bound: int, Umax: int,
                 r_max: int):
        global umax
        global rmax
        self.taille_initial_population = taille_initial_population  # taille initial de la population
        self.max_iter_init = max_iter_init  # nombre d'iterations maximum pour la génération de la population initial
        self.max_iter = max_iter  # nombre d'itération de l'algorithm PFA
        self.bound = bound  # c'est borne qui sert à la génération des solutions (individus) aléatoires
        umax = Umax  # parametre pour adapter PFA au problème du sac à dos
        rmax = r_max  # parametre pour adapter PFA au problème du sac à dos



class Execution:

    def __init__(self):
        self.iterations = list()
        self.evaluations = list()

    def addIteration(self, it):
        self.iterations.append(it)

    def addEvaluation(self, eval):
        self.evaluations.append(eval)

class SolutionCLass:

    def __init__(self):
        self.solution_vector = list()
        self.solution_fitness = list()
        self.solution_time = list()
        self.executions = list()

    def addSolutionVector(self, s_v):
        self.solution_vector.append(s_v)

    def addSolutionFitness(self, s_f):
        self.solution_fitness.append(s_f)

    def addSolutionTime(self, s_t):
        self.solution_time.append(s_t)

    def addExecution(self, exec):
        self.executions.append(exec)

def fitness(S):
    # fonction fitness (fonction d'évaluation de la solution)
    global items
    global capacity
    S = np.array(S)
    if (S < 0).any():
        return -1
    profit = 0
    C = capacity
    for i in range(0, len(S)):
        profit += (items[i][0] * S[i])
        C -= (items[i][1] * S[i])
        if C < 0:
            return -1
    return profit


def init_population(pfa_parameters: PFAParameters):
    global capacity
    global items

    max_iter_init = pfa_parameters.max_iter_init
    population = list()

    while (len(population) < pfa_parameters.taille_initial_population) and (max_iter_init > 0):

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

    return pathfinder, pathfinder_pos


def update_pathfinder(pathfinder_k, pathfinder_k_1, A):
    r3 = [random.randint(0, rmax) for i in range(0, len(items))]
    pathfinder_new = np.add(pathfinder_k, np.multiply(np.multiply(2, r3), np.subtract(pathfinder_k, pathfinder_k_1)))
    pathfinder_new = np.add(pathfinder_new, A)
    return pathfinder_new


def generate_epsilon(K, Kmax, population):
    global items

    u1 = [random.randint(-umax, umax) for i in range(0, len(items))]
    ct = (1 - K / Kmax)
    epsilon = np.multiply(ct, u1)
    i = random.randint(0, len(population) - 1)
    j = random.randint(0, len(population) - 1)
    Dij = np.abs(np.subtract(population[i], population[j]))
    epsilon = np.multiply(epsilon, Dij)
    return epsilon.astype(int)


def update_member(population, i, pathfinder_k, epsilon, alpha, beta):
    global rmax

    new_member = population[i]
    r1 = [random.randint(0, rmax) for i in range(0, len(items))]
    R1 = np.multiply(alpha, r1)
    r2 = [random.randint(0, rmax) for i in range(0, len(items))]
    R2 = np.multiply(beta, r2)
    j = random.randint(0, len(population) - 1)

    while j == i:
        j = random.randint(0, len(population) - 1)

    new_member = np.add(new_member, np.multiply(R1, np.subtract(population[i], population[j])))
    new_member = np.add(new_member, np.multiply(R2, np.subtract(pathfinder_k, population[i])))
    new_member = np.add(new_member, epsilon)
    return new_member


def generateA(Kmax, K):
    global items
    global umax

    u2 = [random.randint(-umax, umax) for i in range(0, len(items))]
    return np.multiply(int(np.exp(-2 * K / Kmax)), u2)


def PFA(pfa_parameters):
    global items
    global capacity

    population = init_population(pfa_parameters)
    pathfinder_k, pathfinder_pos = find_pathfinder(population)
    pathfinder_k_1 = pathfinder_k
    best_fitness = fitness(pathfinder_k)
    K = 1
    A = generateA(K, pfa_parameters.max_iter)
    epsilon = generate_epsilon(K, pfa_parameters.max_iter, population)
    execution = Execution()
    while K < pfa_parameters.max_iter:
        print(f"{K} --> {fitness(pathfinder_k)}")
        execution.addIteration(K)
        execution.addEvaluation(fitness(pathfinder_k))
        alpha = random.randint(1, 2)
        beta = random.randint(1, 2)

        # update path finder position
        new_pathfinder = update_pathfinder(pathfinder_k, pathfinder_k_1, A)
        new_pathfinder_fitness = fitness(new_pathfinder)
        if new_pathfinder_fitness > best_fitness:
            pathfinder_k_1 = pathfinder_k
            pathfinder_k = new_pathfinder
            best_fitness = new_pathfinder_fitness

        new_population = population.copy()
        # update population positions
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
        epsilon = generate_epsilon(K, pfa_parameters.max_iter, population)

    return pathfinder_k, execution


def genererVoisins(S, borne_sup_perturbation):
    voisins = list()
    """
    for i in range(0, len(S)):

        tmp_S = S.copy()
        for k in range(1, borne_sup_perturbation):
            tmp_S[i] += 1
            tmp_to_append = tmp_S.copy()
            voisins.append(tmp_to_append)
        tmp_S = S.copy()
        for k in range(1, borne_sup_perturbation):
            if tmp_S[i] >= 1:
                tmp_S[i] -= 1
                tmp_to_append = tmp_S.copy()
                voisins.append(tmp_to_append)
    """

    for i in range(0, len(S)):
        for k in range(1, borne_sup_perturbation):
            tmp_S = S.copy()
            tmp_S[i] = tmp_S[i] + k
            voisins.append(tmp_S)
        for k in range(1, borne_sup_perturbation):
            tmp_S = S.copy()
            if tmp_S[i] >= k:
                tmp_S[i] -= k
                voisins.append(tmp_S)

    """
    for k in range(0, 100000):
        random_solution = [random.randint(0, borne_sup_perturbation) for i in range(0, len(S))]
        if(fitness(random_solution) <= 0):
            continue
        voisins.append(random_solution)
    """
    return voisins


def recuit_simule(borne_inf_temperature, R, T, S0, borne_sup_perturbation):
    S_etoile = S0.copy()
    S = S0.copy()
    # R = 100
    # borne_inf_temperature = 1
    # T = 10230
    alpha = .9
    stop = False
    K = 0
    exec = Execution()
    while not stop:
        S_etoile_ancienne = S_etoile.copy()
        voisins = genererVoisins(S, borne_sup_perturbation)
        # selectionner le meilleurs des voisins
        exec.addIteration(K)
        exec.addEvaluation(fitness(S_etoile))
        iter = R

        while iter > 0:
            i = random.randint(0, len(voisins) - 1)
            S_prime = voisins[i].copy()
            if (fitness(S_prime) >= fitness(S)):
                S = S_prime
            else:
                r = random.random()
                Delta_F = fitness(S) - fitness(S_prime)
                if r < math.exp(-(Delta_F / T)):
                    S = S_prime

            if fitness(S) > fitness(S_etoile):
                S_etoile = S

            iter -= 1

        T = alpha * T
        K = K + 1
        if T <= borne_inf_temperature:
            stop = True

    print(exec.iterations)
    return S_etoile, exec



def worker(file, pfa_parameters, withHybrid) -> SolutionCLass:
    global items
    global capacity

    print('working..')
    # file = open(file_path, 'r')
    benefices = []
    solutions = SolutionCLass()
    for line in file:
        values = line.split()
        print(values)
        if len(values) == 0:
            continue
        else:
            if values[0] == b'$1':
                capacity = int(values[-1])
            elif values[0] == b'$2':
                benefices = values[1:]
            elif values[0] == b'$3':
                values = values[1:]
                for i in range(0, len(benefices)):
                    items.append([int(benefices[i]), int(values[i]), i])

                # Pathfinder only
                time_start = time.perf_counter()
                S_etoile, pfa_execution = PFA(pfa_parameters)
                time_end = time.perf_counter()
                time_pathfinder = time_end - time_start
                print("Solution S* = " + str(S_etoile))
                print(fitness(S_etoile))
                print(time_end - time_start)
                if withHybrid:
                    # hybridation with recuit_simule
                    time_start = time.perf_counter()
                    S_etoile, execution_r = recuit_simule(1, 1000, 1023000, S_etoile, 6)
                    time_end = time.perf_counter()
                    print("Solution S* = " + str(S_etoile))
                    print(fitness(S_etoile))
                    time_recuit = time_end - time_start
                    total_time = time_recuit = time_pathfinder
                    print(time_end - time_start)
                    solutions.addSolutionTime(total_time)
                    solutions.addSolutionFitness(fitness(S_etoile))
                    solutions.addSolutionVector(S_etoile)
                    iterations_recuit_simule = execution_r.iterations.copy()
                    for i in range(len(iterations_recuit_simule)):
                        iterations_recuit_simule[i] += pfa_parameters.max_iter
                    hybrid_iterations = pfa_execution.iterations + iterations_recuit_simule
                    execution_r.iterations = hybrid_iterations
                    solutions.addExecution(execution_r)
                else:
                    solutions.addSolutionVector(S_etoile)
                    solutions.addSolutionFitness(fitness(S_etoile))
                    solutions.addSolutionTime(time_pathfinder)
                    solutions.addExecution(pfa_execution)

                items = []
                capacity = 0
                umax = None
                rmax = None

    return solutions
