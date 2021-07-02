import math

import numpy as np

from deliver.problem_solver import ProblemSolver
import random


class GeneticSolver(ProblemSolver):
    groups = None
    population = []
    population_size = 25
    random_portion = 0

    def __init__(self, problem, population_size=25, random_portion=0):
        super().__init__(problem)

        self.population_size = population_size
        self.random_portion = random_portion
        self.group_customers()

    def solve(self):
        return

    def group_customers(self):
        self.groups = [[] for i in range(len(self.problem.depots))]
        # Group customers to closest depot
        for c in self.problem.customers:
            depot, depot_index, dist = self.find_closest_depot(c)
            self.groups[depot_index].append(c)
        print(self.groups)

    def find_closest_depot(self, customer_id):
        closest_depot = None
        closest_distance = -1
        for i, depot in enumerate(self.problem.depots):
            from_c_to_d = self.problem.matrix.data[i][customer_id]
            from_d_to_c = self.problem.matrix.data[customer_id][i]
            total_distance = from_c_to_d + from_d_to_c
            if closest_depot is None or total_distance < closest_distance:
                closest_depot = (depot, i)
                closest_distance = total_distance

        return closest_depot[0], closest_depot[1], closest_distance
