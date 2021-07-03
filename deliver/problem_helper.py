import json
from cmath import inf

from deliver.customer import Customer
from deliver.depot import Depot
from deliver.job import Job
from deliver.matrix import Matrix
from deliver.vehicle import Vehicle


class ProblemHelper:
    data = None

    def __init__(self, json_path):
        self.data = self.read_json(json_path)
        self.vehicles, self.jobs, \
        self.matrix, self.depots, self.customers = self.get()

    @staticmethod
    def read_json(file_path):
        """
        Returns JSON object as a dictionary
        @param file_path: json file path
        @return: JSON object
        """
        f = open(file_path, )
        return json.load(f)

    def get(self):
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
        return [Depot(d.start_index, 2, inf, d.capacity[0]) for d in vehicles]
        # n_vehicles_in_depot = Counter(depot_idxs)
        # return [Depot(d, inf, inf, d) for d in list(set(depot_idxs))]

    @staticmethod
    def define_customers(jobs):
        return [Customer(index, j.service, j.delivery[0], j.location_index) for index, j in enumerate(jobs)]
