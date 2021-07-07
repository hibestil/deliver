import math
import random

import numpy as np

from deliver.genetic_algorithm.problem_solver import ProblemSolver
from deliver.problem.customer import Customer


class GeneticSolver(ProblemSolver):
    """A class used to represent a Genetic Algorithm based VRP Problem Solver"""
    groups = None
    population = []

    def __init__(self, problem,
                 random_portion=0,
                 generations=2500,
                 population_size=50,
                 crossover_rate=0.60,
                 heuristic_mutate_rate=0.20,
                 inversion_mutate_rate=0.25,
                 depot_move_mutate_rate=0.25,
                 best_insertion_mutate_rate=0.1,
                 route_merge_rate=0.05,
                 ):
        super().__init__(problem)
        # Genetic Algorithm Parameters
        self.population_size = population_size
        self.random_portion = random_portion
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.heuristic_mutate_rate = heuristic_mutate_rate
        self.inversion_mutate_rate = inversion_mutate_rate
        self.depot_move_mutate_rate = depot_move_mutate_rate
        self.best_insertion_mutate_rate = best_insertion_mutate_rate
        self.route_merge_rate = route_merge_rate
        # Initialize variables
        self.group_customers()
        self.initialize_population()

    def solve(self, log=True, intermediate_prints=True):
        """
        Solves the defined vrp problem
        Args:
            log: Terminal logging option. Shows logs if True
            intermediate_prints: Option for printing intermediate route solutions

        Returns:
            Best solution
        """
        # Iterate generations
        for g in range(self.generations):
            # Log in every 10 generation
            if log and g % 10 == 0:
                best = max(self.population, key=lambda x: x[1])
                print('[Generation {}] Best score: {} Consistent: {}'
                      .format(g, best[1], self.is_consistent(best[0])))
            # Print intermediate route plans in every 100 generation
            if intermediate_prints and g % 100 == 0:
                self.population.sort(key=lambda x: -x[1])
                self.show(self.population[0][0])
            # Do selection
            selection = self.select(
                self.heuristic_mutate_rate + self.inversion_mutate_rate
                + self.crossover_rate + self.depot_move_mutate_rate
                + self.best_insertion_mutate_rate
                + self.route_merge_rate)
            # Map and Convert selection to list
            selection = list(map(lambda x: x[0], selection))

            # Apply Crossover
            offset = 0
            for i in range(int((self.population_size * self.crossover_rate) / 2)):
                p1, p2 = selection[2 * i + offset], selection[
                    2 * i + 1 + offset]
                self.crossover(p1, p2)
                self.crossover(p2, p1)
            offset += int(self.population_size * self.crossover_rate)

            # Apply Heuristic Mutate
            for i in range(int(self.population_size * self.heuristic_mutate_rate)):
                self.heuristic_mutate(selection[i + offset])
            offset += int(self.population_size * self.heuristic_mutate_rate)

            # Apply Inversion Mutate
            for i in range(int(self.population_size * self.inversion_mutate_rate)):
                self.inversion_mutate(selection[i + offset])
            offset += int(self.population_size * self.inversion_mutate_rate)

            # Apply Depot Mutate
            for i in range(int(self.population_size * self.depot_move_mutate_rate)):
                self.depot_move_mutate(selection[i + offset])
            offset += int(self.population_size * self.depot_move_mutate_rate)

            # Apply Insertion Mutate
            for i in range(int(self.population_size * self.best_insertion_mutate_rate)):
                self.best_insertion_mutate(selection[i + offset])
            offset += int(self.population_size * self.best_insertion_mutate_rate)

            # Apply Route Merge
            for i in range(int(self.population_size * self.route_merge_rate)):
                self.route_merge(selection[i + offset])
            offset += int(self.population_size * self.route_merge_rate)

            # Select population
            self.population = self.select(1.0, elitism=4)

        # Sort Population
        self.population.sort(key=lambda x: -x[1])
        print("\n\nFinished training")

        best_solution = None
        # Check consistency of the first population instance
        if self.is_consistent(self.population[0][0]):
            print('Best score: {}, best distance: {}'.format(
                self.population[0][1],
                self.evaluate(self.population[0][0], True)))
            # Select as best solution
            best_solution = self.population[0][0]
        else:
            # Check all population instance consistencies while found a consistent one.
            for c in self.population:
                if self.is_consistent(c[0]):
                    print('Best score: {}, best distance: {}'.format(
                        c[1], self.evaluate(c[0], True)))
                    best_solution = c[0]
                    break
            else:
                print('Found no consistent solutions.')

        return best_solution

    def group_customers(self):
        """
        Group customers by proximity to depots
        """
        self.groups = [[] for i in range(len(self.problem.depots))]
        # Group customers to closest depot
        for c in self.problem.customers:
            depot, depot_index, dist = self.find_closest_depot(c)
            self.groups[depot_index].append(c.id)

    def find_closest_depot(self, customer):
        """
        Find closest depot to specified Customer
        """
        closest_depot = None
        closest_distance = -1
        for i, depot in enumerate(self.problem.depots):
            from_c_to_d = self.distance(depot, customer)
            from_d_to_c = self.distance(customer, depot)
            min_distance = from_c_to_d + from_d_to_c
            if closest_depot is None or min_distance < closest_distance:
                closest_depot = (depot, i)
                closest_distance = min_distance

        return closest_depot[0], closest_depot[1], closest_distance

    def distance(self, source, destination):
        """
        Gives the distance between given Customer and Depot objects.
        """
        # Correct location index if one of source/destination is a Customer instance.
        s_id = source.id
        d_id = destination.id
        if isinstance(source, Customer):
            s_id = source.location_index
        if isinstance(destination, Customer):
            d_id = destination.location_index

        return self.problem.matrix.data[s_id][d_id]

    def show(self, chromosome):
        """
        Print out the route plan for given chromosome
        Args
            chromosome: Chromosome of the route solution
        """
        routes = self.decode(chromosome)
        total_duration = self.evaluate(chromosome, True)

        print("-----------------------SUMMARY-----------------------")
        print("Total duration : {} ".format(total_duration))

        for d, depot in enumerate(self.problem.depots):
            for r, route in enumerate(routes[d]):
                if route:
                    self.print_route_info(d, route, depot)

    def print_route_info(self, d, route, depot):
        """
        Prints vehicle and route plan
        Args:
            d: current depot index
            route: a list of customers in route array
            depot: current depot object
        """
        # Calculate given route length and load
        route_length, route_load = self.evaluate_route(route, depot, True)

        customers = self.problem.customers
        end_depot = self.find_closest_depot(customers[route[-1]])[1]

        print("----------------------------------------------------")
        print("Vehicle :", self.problem.depots[d].id)
        print("\t|_ Leaves from depot", self.problem.depots[d].id)
        print("\t|_ Amount of carried load by this vehicle is : ", route_load)
        print("\t|_ Goes to these customers respectively : ")
        for c in route:
            print("\t\t|_ customer: {}\tdemand:{}".format(
                customers[c].location_index,
                customers[c].demand))
        print("\t|_ Vehicle returns to the depot", end_depot)
        print("\t|_ Total duration of this trip is ", route_length)

    def create_output_json(self, chromosome):
        """
        Create expected output json.
        Args:
            chromosome: given solution chromosome

        Returns:
            JSON formatted output object.
        """
        routes = self.decode(chromosome)
        total_duration = self.evaluate(chromosome, True)

        output = {}
        output["total_delivery_duration"] = str(total_duration)
        routes_json = {}
        counter = 1
        for d, depot in enumerate(self.problem.depots):
            for r, route in enumerate(routes[d]):
                if route:
                    route_json = {}
                    route_length, route_load = self.evaluate_route(route, depot, True)
                    customers = self.problem.customers
                    jobs = []
                    for c in route:
                        jobs.append(str(customers[c].id))
                    route_json["jobs"] = jobs
                    route_json["delivery_duration"] = str(route_length)
                    routes_json[str(counter)] = route_json
                    counter = counter + 1
        output["routes"] = routes_json
        return output

    def encode(self, routes):
        """
        Creates chromosome from routes array
        Example :   routes = [[[1, 2, 5, 6, 4, 0]], [], [[3]]]
                    chromosome = [1, 2, 5, 6, 4, 0, -1, -1, 3]
        Args:
            routes: given routes array
        Returns:
            chromosome formatted routes array
        """
        chromosome = []
        for d in range(len(routes)):
            if d != 0:
                chromosome.append(-1)
            for r in range(len(routes[d])):
                if r != 0:
                    chromosome.append(0)
                chromosome.extend(routes[d][r])
        return chromosome

    def decode(self, chromosome):
        """
        Creates routes array from chromosome
        Example :   chromosome = [1, 2, 5, 6, 4, 0, -1, -1, 3]
                    routes = [[[1, 2, 5, 6, 4, 0]], [], [[3]]]
        Args:
            chromosome: given chromosome
        Returns:
            routes
        """
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
        """
        Schedules the route. If route is empty creates a new route
        Args:
            route:

        Returns:
            Scheduled route
        """
        if not len(route):
            return route
        new_route = []
        prev_cust = random.choice(route)
        route.remove(prev_cust)
        new_route.append(prev_cust)

        while len(route):
            prev_cust = min(route, key=lambda x: self.distance(
                self.problem.customers[x - 1],
                self.problem.customers[prev_cust - 1]))
            route.remove(prev_cust)
            new_route.append(prev_cust)
        return new_route

    def is_consistent(self, chromosome):
        """
        Checks consistency of given chromosome
        Args:
            chromosome:

        Returns:
            Boolean bonsistency value

        """
        for c in self.problem.customers:
            if c.id not in chromosome:
                return False

        routes = self.decode(chromosome)
        for d in range(len(routes)):
            depot = self.problem.depots[d]
            if len(routes[d]) > depot.max_vehicles:
                return False
            for route in routes[d]:
                if not self.is_consistent_route(route, depot):
                    return False

        return True

    def is_consistent_route(self, route, depot, include_reason=True):
        """
        Checks consistency of given route
        Args:
            route: The route that the vehicle will trip
            depot: The depot (vehicle)
            include_reason: Flag for restriction applications

        Returns:
            Consistency value of given route-depot pair
        """
        route_load = 0
        route_duration = 0
        last_pos = depot
        for c in route:
            customer = self.problem.customers[c]
            route_load += customer.demand
            route_duration += self.distance(last_pos, customer) \
                              + customer.service_duration
            last_pos = customer
        route_duration += self.find_closest_depot(last_pos)[2]

        if include_reason:

            if route_load > depot.max_load:
                return False, 1
            if depot.max_duration != 0 and route_duration > depot.max_duration:
                return False, 2
            return True, 0
        return route_load <= depot.max_load and (
            depot.max_duration == 0 or route_duration <= depot.max_duration)

    def initialize_population(self):
        """
        Initialize the population.
        Here we're creating chromosomes by using heuristic and random chromosome generations.
        """
        for x in range(int(self.population_size * (1 - self.random_portion))):
            chromosome = self.create_heuristic_chromosome(self.groups)
            self.population.append((chromosome, self.evaluate(chromosome)))

        for x in range(int(self.population_size * self.random_portion)):
            chromosome = self.create_random_chromosome(self.groups)
            self.population.append((chromosome, self.evaluate(chromosome)))

    def create_heuristic_chromosome(self, groups):
        """
        Creates chromosomes by using heuristic way.
        Args:
            groups: initial depot-customer pairs. i.e = [[1,2],[3,6],[4,5,7]]

        Returns:
            Heuristically created chromosome
        """
        depots = self.problem.depots
        customers = self.problem.customers
        # Group customers in routes according to savings
        routes = [[] for i in range(len(depots))]
        missing_customers = list(map(lambda x: x.id, customers))
        for d in range(len(groups)):
            depot = depots[d]
            savings = []
            for i in range(len(groups[d])):
                ci = customers[groups[d][i]]
                savings.append([])
                for j in range(len(groups[d])):
                    if j <= i:
                        savings[i].append(0)
                    else:
                        cj = customers[groups[d][j]]
                        savings[i].append(self.distance(depot, ci) + self.distance(depot, cj) -
                                          self.distance(ci, cj))
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
                        if ci == cj:
                            route = [ci]
                        else:
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

    def create_random_chromosome(self, groups):
        """
        Creates chromosomes by using random way.
        Args:
            groups: initial depot-customer pairs. i.e = [[1,2],[3,6],[4,5,7]]

        Returns:
            Randomly created chromosome
        """
        routes = []
        for d in range(len(groups)):
            depot = self.problem.depots[d]
            group = groups[d][:]
            random.shuffle(group)
            routes.append([[]])

            r = 0
            route_cost = 0
            route_load = 0
            last_pos = depot
            for c in group:
                customer = self.problem.customers[c]
                cost = self.distance(last_pos, customer) \
                       + customer.service_duration + \
                       self.find_closest_depot(customer)[2]
                if route_cost + cost > depot.max_duration or route_load + customer.demand > depot.max_load:
                    r += 1
                    routes[d].append([])
                routes[d][r].append(c)

        return self.encode(routes)

    def crossover(self, p1, p2):
        protochild = [None] * max(len(p1), len(p2))
        cut1 = int(random.random() * len(p1))
        cut2 = int(cut1 + random.random() * (len(p1) - cut1))
        substring = p1[cut1:cut2]

        for i in range(cut1, cut2):
            protochild[i] = p1[i]

        p2_ = list(reversed(p2))
        for g in substring:
            if g in p2_:
                p2_.remove(g)
        p2_.reverse()

        j = 0
        for i in range(len(protochild)):
            if protochild[i] is None:
                if j >= len(p2_):
                    break
                protochild[i] = p2_[j]
                j += 1

        i = len(protochild) - 1
        while protochild[i] is None:
            protochild.pop()
            i -= 1

        self.population.append((protochild, self.evaluate(protochild)))

    def evaluate(self, chromosome, return_distance=False):
        """
        The chromosome cost function implementation for proposed genetic algorithm.
        Args:
            chromosome: The chromosome that needs to be evaluated
            return_distance: Returns distance if the value is True

        Returns:
            Distance or score (depending on return_distance option value)
        """
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
                route_length, route_load = self.evaluate_route(route, depot,
                                                               True)

                score += route_length

                if depot.max_duration and route_length > depot.max_duration:
                    score += (route_length - depot.max_duration) * 20
                if route_load > depot.max_load:
                    score += (route_load - depot.max_load) * 50
        if return_distance:
            return score
        return 1 / score

    def evaluate_route(self, route, depot, return_load=False):
        """
       The route cost function implementation for proposed genetic algorithm.
        Args:
            route: The route that needs to be evaluated.
            depot: The depot that needs to be evaluated.
            return_load: If evaluated load needed, this value should set True

        Returns:
            Route length and load
        """
        if len(route) == 0:
            if return_load:
                return 0, 0
            return 0
        route_load = 0
        route_length = 0
        customer = None
        last_pos = depot
        for cid in route:
            customer = self.problem.customers[cid]
            route_load += customer.demand
            route_length += self.distance(last_pos, customer)
            route_length += customer.service_duration
            last_pos = customer
        route_length += self.find_closest_depot(customer)[1]

        if return_load:
            return route_length, route_load
        return route_length

    def select(self, portion, elitism=0):
        """
        Apply selection operation on population.
        Args:
            portion:
            elitism:

        Returns:
            Selected portion
        """
        total_fitness = sum(map(lambda x: x[1], self.population))
        weights = list(
            map(lambda x: (total_fitness - x[1]) / (
                total_fitness * (self.population_size - 1)),
                self.population))
        selection = random.choices(self.population, weights=weights, k=int(
            self.population_size * portion - elitism))
        self.population.sort(key=lambda x: -x[1])
        if elitism > 0:
            selection.extend(self.population[:elitism])
        return selection

    def heuristic_mutate(self, p):
        """
        Applies heuristic mutate operation on given population.
        Args:
            p: population
        """
        g = []
        for i in range(3):
            g.append(int(random.random() * len(p)))

        offspring = []
        for i in range(len(g)):
            for j in range(len(g)):
                if g == j:
                    continue
                o = p[:]
                o[g[i]], o[g[j]] = o[g[j]], o[g[i]]
                offspring.append((o, self.evaluate(o)))

        selected_offspring = max(offspring, key=lambda o: o[1])
        self.population.append(selected_offspring)

    def inversion_mutate(self, p):
        """
        Applies inversion mutate operation on given population.
        Args:
            p: population
        """
        cut1 = int(random.random() * len(p))
        cut2 = int(cut1 + random.random() * (len(p) - cut1))

        if cut1 == cut2:
            return
        if cut1 == 0:
            child = p[:cut1] + p[cut2 - 1::-1] + p[cut2:]
        else:
            child = p[:cut1] + p[cut2 - 1:cut1 - 1:-1] + p[cut2:]
        self.population.append((child, self.evaluate(child)))

    def best_insertion_mutate(self, p):
        """
        Applies best insertion mutate operation on given population.
        Args:
            p: population
        """
        g = int(random.random() * len(p))

        best_child = None
        best_score = 0

        for i in range(len(p) - 1):
            child = p[:]
            gene = child.pop(g)
            child.insert(i, gene)
            score = self.evaluate(child)
            if score > best_score:
                best_score = score
                best_child = child

        self.population.append((best_child, best_score))

    def depot_move_mutate(self, p):
        """
        Applies depot move mutate operation on given population.
        Args:
            p: population
        """
        if -1 not in p:
            return
        i = int(random.random() * len(p))
        while p[i] != -1:
            i = (i + 1) % len(p)

        move_len = int(random.random() * 10) - 5
        new_pos = (i + move_len) % len(p)

        child = p[:]
        child.pop(i)
        child.insert(new_pos, -1)
        self.population.append((child, self.evaluate(child)))

    def route_merge(self, p):
        """
        Merges route of given population chromosome p.
        Args:
            p: population
        """
        routes = self.decode(p)

        d1 = int(random.random() * len(routes))
        r1 = int(random.random() * len(routes[d1]))
        d2 = int(random.random() * len(routes))
        r2 = int(random.random() * len(routes[d2]))

        if random.random() < 0.5:
            limit = int(random.random() * len(routes[d2][r2]))
        else:
            limit = len(routes[d2][r2])

        reverse = random.random() < 0.5

        for i in range(limit):
            if reverse:
                routes[d1][r1].append(routes[d2][r2].pop(0))
            else:
                routes[d1][r1].append(routes[d2][r2].pop())
        routes[d1][r1] = self.schedule_route(routes[d1][r1])
        routes[d2][r2] = self.schedule_route(routes[d2][r2])
        child = self.encode(routes)
        self.population.append((child, self.evaluate(child)))
