import numpy as np
import random
import math
import time

items = []
capacity = 0

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
    i = random.randint(0, len(S)-1)
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
    return voisins

def recuit_simule(borne_inf_temperature, R, T, S0, borne_sup_perturbation):

    S_etoile = S0.copy()
    S = S0.copy()
    #R = 100
    #borne_inf_temperature = 1
    #T = 10230
    alpha = .9
    stop = False

    while not stop:

        S_etoile_ancienne = S_etoile.copy()
        voisins = genererVoisins(S,borne_sup_perturbation)
        #selectionner le meilleurs des voisins
        iter = R

        while iter > 0:

            i = random.randint(0, len(voisins)-1)
            S_prime = voisins[i]
            if(fitness(S_prime) >=  fitness(S)):
                S = S_prime
            else:
                r = random.random()
                Delta_F = fitness(S)-fitness(S_prime)
                if r < math.exp(-(Delta_F/T)):
                    S = S_prime

            if fitness(S) > fitness(S_etoile):
                S_etoile = S

            iter -= 1

        T = alpha*T
        if T <= borne_inf_temperature:
            stop = True

    return S_etoile

def main():
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

                S_etoile = [0]*19
                time_start = time.perf_counter()
                S_etoile = recuit_simule(1,10000,102300,S_etoile,15)
                time_end = time.perf_counter()
                print("Solution S* = " + str(S_etoile))
                print(fitness(S_etoile))
                print(time_end - time_start)

if __name__ == '__main__':
    main()
