import json

from deliver.job import Job
from deliver.matrix import Matrix
from deliver.vehicle import Vehicle


class ProblemHelper:
    data = None

    def __init__(self, json_path):
        self.data = self.read_json(json_path)

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
        else:
            raise Exception("Json file is not provided")
        return vehicles, jobs, matrix
