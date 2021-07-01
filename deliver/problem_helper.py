import json

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

    def define_depots(self, vehicles):
        depot_idxs = [d.start_index for d in vehicles]
        return list(set(depot_idxs))

    def define_customers(self, jobs):
        customer_idxs = [j.location_index for j in jobs]
        return list(set(customer_idxs))
