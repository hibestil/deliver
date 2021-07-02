class Depot:

    def __init__(self, idx, max_vehicles, max_duration, max_load):
        self.id = idx
        self.max_vehicles = max_vehicles
        self.max_duration = max_duration
        self.max_load = max_load
        self.closest_customers = []
