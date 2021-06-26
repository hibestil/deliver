class Vehicle:
    vehicle_id = None  # Vehicle id
    start_index = None  # Index of the vehicle’s starting location
    capacity = 0  # Initial carboy capacity of the vehicle

    def __init__(self, id, start_index, capacity):
        self.vehicle_id = id
        self.start_index = start_index
        self.capacity = capacity
