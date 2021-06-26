import json


class DatasetHelper:
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
