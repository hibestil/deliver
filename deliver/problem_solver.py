class ProblemSolver:
    problem = None
    vehicles = None
    jobs = None
    matrix = None

    def __init__(self, problem):
        self.problem = problem
        self.vehicles, self.jobs, self.matrix = problem.get()
