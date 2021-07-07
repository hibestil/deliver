import json
import math
from cmath import inf

import numpy as np

from deliver.problem.customer import Customer
from deliver.problem.depot import Depot
from deliver.problem.job import Job
from deliver.problem.matrix import Matrix
from deliver.problem.vehicle import Vehicle


class ProblemHelper:
    """
    Problem helper class allows user to read and structure the input files.
    """
    data = None

    def __init__(self, file_path, benchmark=False):
        """
        Init class
        Args:
            file_path: Input file path
            benchmark: Defines the type of input file format.
                          If json input will be used this value is False,
                          else if Cordeau’s Instances will be used it must be True
        """
        if not benchmark:
            self.data = self.read_json(file_path)
            self.vehicles, self.jobs, \
            self.matrix, self.depots, self.customers = self.get()
        else:
            self.vehicles, \
            self.matrix, self.depots, self.customers = self.get_from_benchmark(file_path)

    @staticmethod
    def read_json(file_path):
        """
        Returns JSON object as a dictionary
        Args:
            file_path: json file path

        Returns:
            JSON object
        """
        f = open(file_path, )
        return json.load(f)

    def get_from_benchmark(self, path):
        """
        Reads benchmarking set (Cordeau’s Instances) to use in algorithm.
        References :
            - [1] http://neo.lcc.uma.es/vrp/vrp-instances/description-for-files-of-cordeaus-instances/
            - [2] https://github.com/fboliveira/MDVRP-Instances
        Args:
            path: Benchmark input file path
        Returns:
             vehicles, m, depots, customers data
        """
        depots = []
        customers = []
        with open(path) as f:
            max_vehicles, num_customers, num_depots = tuple(map(lambda z: int(z), f.readline().strip().split()))

            for i in range(num_depots):
                max_duration, max_load = tuple(map(lambda z: int(z), f.readline().strip().split()))
                depots.append(Depot(i - 1, max_vehicles, max_duration, max_load))

            for i in range(num_customers):
                vals = tuple(map(lambda z: int(z), f.readline().strip().split()))
                cid, x, y, service_duration, demand = (vals[j] for j in range(5))
                customers.append(Customer(cid - 1, service_duration, demand, num_depots + i))
                customers[i].pos = (x, y)
            for i in range(num_depots):
                vals = tuple(map(lambda z: int(z), f.readline().strip().split()))
                cid, x, y = (vals[j] for j in range(3))
                depots[i].pos = (x, y)

            # Create matrix
            matrix = np.zeros((num_depots + num_customers, num_depots + num_customers))
            # Create depots portion of matrix
            for i in range(num_depots):
                for j in range(num_depots):
                    if i != j:
                        matrix[i][j] = self.point_distance(depots[i].pos, depots[j].pos)
            # Create customers portion of matrix
            for i in range(num_customers):
                for j in range(num_customers):
                    if i != j:
                        matrix[num_depots + i][num_depots + j] = self.point_distance(customers[i].pos, customers[j].pos)
            for i in range(num_depots):
                for j in range(num_customers):
                    if i != j:
                        matrix[i][num_depots + j] = self.point_distance(depots[i].pos, customers[j].pos)
            for i in range(num_customers):
                for j in range(num_depots):
                    if i != j:
                        matrix[num_depots + i][j] = self.point_distance(customers[i].pos, depots[j].pos)

            # Create vehicles from depots
            vehicles = [Vehicle(d.id, d.id, d.max_load) for d in depots]

            m = Matrix(matrix)
        return vehicles, m, depots, customers

    @staticmethod
    def point_distance(p1, p2):
        """
        Measure euclidean distance two points p1 and p2.
        Args:
            p1: First point
            p2: Second point
        Returns:
             Distance between points
        """
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    def get(self):
        """
        Allows to get structured data from object
        Returns:
             vehicles, jobs, matrix, depots, customers
        """
        if self.data:
            vehicles = [Vehicle(**v) for v in self.data["vehicles"]]
            jobs = [Job(**j) for j in self.data["jobs"]]
            matrix = Matrix(self.data["matrix"])
            depots = self.define_depots(vehicles)
            customers = self.define_customers(jobs)
        else:
            raise Exception("Json file is not provided")
        return vehicles, jobs, matrix, depots, customers

    @staticmethod
    def define_depots(vehicles):
        """
        Converts vehicle objects to depot instances.
        Args:
            vehicles: List of vehicle objects
        Returns:
             List of depot objects
        """
        return [Depot(d.start_index, 2, inf, d.capacity[0]) for d in vehicles]
        # n_vehicles_in_depot = Counter(depot_idxs)
        # return [Depot(d, inf, inf, d) for d in list(set(depot_idxs))]

    @staticmethod
    def define_customers(jobs):
        """
        Converts job objects to customer instances.
        Returns:
             List of Customer objects
        """
        return [Customer(index, j.service, j.delivery[0], j.location_index) for index, j in enumerate(jobs)]
