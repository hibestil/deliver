class Solve:

    def __init__(self, matrix, jobs, vehicles, maximum_route_duration=None, minimize_k=True):
        self.minimize_k = minimize_k
        self.jobs = jobs
        self.matrix = matrix
        self.vehicles = vehicles
