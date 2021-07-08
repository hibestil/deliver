class Depot:
    """A class used to represent a Depot object"""

    def __init__(self, idx, max_vehicles, max_duration, max_load):
        self.id = idx
        self.max_vehicles = max_vehicles
        self.max_duration = max_duration
        self.max_load = max_load
        self.closest_customers = []

    def __str__(self):
        return str([self.id,
                    self.max_vehicles,
                    self.max_duration,
                    self.max_load,
                    self.closest_customers])
