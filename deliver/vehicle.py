class Vehicle:
    vehicle_id = None
    start_index = None
    capacity = 0

    def __init__(self, vehicle_id, start_index, capacity):
        self.vehicle_id = vehicle_id
        self.start_index = start_index
        self.capacity = capacity
