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
        self.init_population()

    def solve(self):
        return

    def group_customers(self):
        self.groups = [[] for i in range(len(self.problem.depots))]
        # Group customers to closest depot
        for c in self.problem.customers:
            depot, depot_index, dist = self.find_closest_depot(c)
            self.groups[depot_index].append(c.id)
        print(self.groups)

    def find_closest_depot(self, customer):
        closest_depot = None
        closest_distance = -1
        for i, depot in enumerate(self.problem.depots):
            from_c_to_d = self.distance(depot.id, customer.id)
            from_d_to_c = self.distance(customer.id, depot.id)
            min_distance = min(from_c_to_d, from_d_to_c)
            if closest_depot is None or min_distance < closest_distance:
                closest_depot = (depot, i)
                closest_distance = min_distance

        return closest_depot[0], closest_depot[1], closest_distance

    def init_population(self):
        for z in range(int(self.population_size * (1 - self.random_portion))):
            chromosome = self.create_heuristic_chromosome(self.groups)
            self.population.append((chromosome, self.evaluate(chromosome)))

    def distance(self, source, destination):
        # print(source,destination)
        return self.problem.matrix.data[source][destination]

    @staticmethod
    def encode(routes):
        chromosome = []
        for d in range(len(routes)):
            if d != 0:
                chromosome.append(-1)
            for r in range(len(routes[d])):
                if r != 0:
                    chromosome.append(0)
                chromosome.extend(routes[d][r])
        return chromosome

    @staticmethod
    def decode(chromosome):
        routes = [[[]]]
        d = 0
        r = 0
        for i in chromosome:
            if i < 0:
                routes.append([[]])
                d += 1
                r = 0
            elif i == 0:
                routes[d].append([])
                r += 1
            else:
                routes[d][r].append(i)
        return routes

    def schedule_route(self, route):
        if not len(route):
            return route
        new_route = []
        prev_cust = random.choice(route)
        route.remove(prev_cust)
        new_route.append(prev_cust)

        while len(route):
            prev_cust = min(route, key=lambda x: self.distance(self.problem.customers[x - 1].id,
                                                               self.problem.customers[prev_cust - 1].id))
            route.remove(prev_cust)
            new_route.append(prev_cust)
        return new_route

    def is_consistent_route(self, route, depot, include_reason=False):
        route_load = 0
        route_duration = 0
        last_pos = depot
        for c in route:
            customer = self.problem.customers[c - 1]
            route_load += customer.demand
            route_duration += self.distance(last_pos.id, customer.id) + customer.service_duration
            last_pos = customer
        route_duration += self.find_closest_depot(last_pos)[2]

        if include_reason:
            if route_load > depot.max_load:
                return False, 1
            if depot.max_duration != 0 and route_duration > depot.max_duration:
                return False, 2
            return True, 0
        return route_load <= depot.max_load and (depot.max_duration == 0 or route_duration <= depot.max_duration)

    def create_heuristic_chromosome(self, groups):
        depots = self.problem.depots
        customers = self.problem.customers
        # Group customers in routes according to savings
        routes = [[] for i in range(len(depots))]
        missing_customers = list(map(lambda x: x.id, customers))
        for d in range(len(groups)):
            depot = depots[d]
            savings = []
            for i in range(len(groups[d])):
                ci = customers[groups[d][i] - 1]
                savings.append([])
                for j in range(len(groups[d])):
                    if j <= i:
                        savings[i].append(0)
                    else:
                        cj = customers[groups[d][j] - 1]
                        savings[i].append(self.distance(depot.id, ci.id) + self.distance(depot.id, cj.id) -
                                          self.distance(ci.id, cj.id))
            savings = np.array(savings)
            order = np.flip(np.argsort(savings, axis=None), 0)

            for saving in order:
                i = saving // len(groups[d])
                j = saving % len(groups[d])

                ci = groups[d][i]
                cj = groups[d][j]

                ri = -1
                rj = -1
                for r, route in enumerate(routes[d]):
                    if ci in route:
                        ri = r
                    if cj in route:
                        rj = r

                route = None
                if ri == -1 and rj == -1:
                    if len(routes[d]) < depot.max_vehicles:
                        route = [ci, cj]
                elif ri != -1 and rj == -1:
                    if routes[d][ri].index(ci) in (0, len(routes[d][ri]) - 1):
                        route = routes[d][ri] + [cj]
                elif ri == -1 and rj != -1:
                    if routes[d][rj].index(cj) in (0, len(routes[d][rj]) - 1):
                        route = routes[d][rj] + [ci]
                elif ri != rj:
                    route = routes[d][ri] + routes[d][rj]

                if route:
                    if self.is_consistent_route(route, depot, True)[1] == 2:
                        route = self.schedule_route(route)
                    if self.is_consistent_route(route, depot):
                        if ri == -1 and rj == -1:
                            routes[d].append(route)
                            missing_customers.remove(ci)
                            if ci != cj:
                                missing_customers.remove(cj)
                        elif ri != -1 and rj == -1:
                            routes[d][ri] = route
                            missing_customers.remove(cj)
                        elif ri == -1 and rj != -1:
                            routes[d][rj] = route
                            missing_customers.remove(ci)
                        elif ri != -1 and rj != -1:
                            if ri > rj:
                                routes[d].pop(ri)
                                routes[d].pop(rj)
                            else:
                                routes[d].pop(rj)
                                routes[d].pop(ri)
                            routes[d].append(route)

        # Order customers within routes
        for i, depot_routes in enumerate(routes):
            for j, route in enumerate(depot_routes):
                new_route = self.schedule_route(route)
                routes[i][j] = new_route

        chromosome = self.encode(routes)
        chromosome.extend(missing_customers)
        return chromosome

    def evaluate(self, chromosome, return_distance=False):
        for c in self.problem.customers:
            if c.id not in chromosome:
                if return_distance:
                    return math.inf
                return 0

        routes = self.decode(chromosome)
        score = 0
        for depot_index in range(len(routes)):
            depot = self.problem.depots[depot_index]
            for route in routes[depot_index]:
                route_length, route_load = self.evaluate_route(route, depot, True)

                score += route_length

                if depot.max_duration and route_length > depot.max_duration:
                    score += (route_length - depot.max_duration) * 20
                if route_load > depot.max_load:
                    score += (route_load - depot.max_load) * 50
        if return_distance:
            return score
        return 1 / score

    def evaluate_route(self, route, depot, return_load=False):
        if len(route) == 0:
            if return_load:
                return 0, 0
            return 0
        route_load = 0
        route_length = 0
        customer = None
        last_pos = depot.id
        for cid in route:
            customer = self.problem.customers[cid - 1]
            route_load += customer.demand
            route_length += self.distance(last_pos, customer.id)
            route_length += customer.service_duration
            last_pos = customer.id
        route_length += self.find_closest_depot(customer)[1]

        if return_load:
            return route_length, route_load
        return route_length
