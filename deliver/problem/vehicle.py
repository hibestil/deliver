class Vehicle:
    vehicle_id = None  # Vehicle id
    start_index = None  # Index of the vehicle’s starting location
    capacity = 0  # Initial carboy capacity of the vehicle

    def __init__(self, id, start_index, capacity):
        self.vehicle_id = id
        self.start_index = start_index
        self.capacity = capacity

    def __str__(self):
        str = "[Vehicle] :"
        str += "\tid:{}".format(self.vehicle_id)
        str += "\tstart_index:{}".format(self.start_index)
        str += "\tcapacity:{}".format(self.capacity)
        return str
