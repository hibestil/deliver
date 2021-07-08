from deliver.genetic_algorithm.problem_solver import ProblemSolver
from deliver.genetic_algorithm.utils_brute_force import mapdistr, permutate
from deliver.problem.job import Job
from deliver.problem.vehicle import Vehicle


class BruteForceSolver(ProblemSolver):
    """A class used to represent a Brute Force Algorithm based VRP Problem Solver"""

    def __init__(self, problem):
        self.problem = problem

        self.route_combs = mapdistr(self.problem.jobs, len(self.problem.vehicles))
        self.vehicle_perm = permutate(self.problem.vehicles)

    def solve(self, intermediate_prints=False):
        best_v_comb = None
        best_r_comb = None
        min_cost = float('inf')
        counter = 0
        for v in self.vehicle_perm:
            for r in self.route_combs:
                cost = self.calculate_cost(self.problem.matrix.data, v, r)
                counter = counter + 1
                if cost < min_cost and self.is_consistent(v, r):
                    # print(best_v_comb, best_r_comb, min_cost)
                    min_cost = cost
                    best_v_comb = v
                    best_r_comb = r

        return (best_v_comb, best_r_comb, min_cost)

    def create_output_json(self, solution):
        """
        Create expected output json.
        Args:
            solution: given route-vehicle solution

        Returns:
            JSON formatted output object.
        """
        vehicles, routes, total_duration = solution
        output = {"total_delivery_duration": str(total_duration)}
        routes_json = {}
        counter = 1
        for v, vehicle in enumerate(vehicles):
            route_json = {}
            route_length = self.calculate_cost(self.problem.matrix.data,
                                               [vehicle], routes[v],
                                               vehicle_cost=True)
            jobs = []
            for j, job in enumerate(routes[v]):
                jobs.append(str(job.job_id))
            route_json["jobs"] = jobs
            route_json["delivery_duration"] = str(route_length)
            routes_json[str(counter)] = route_json
            counter = counter + 1
        output["routes"] = routes_json
        return output

    @staticmethod
    def get_matrix_index(instance):
        """
        Determines the distance matrix index of different type of objects
        Args:
            instance:

        Returns:

        """
        if isinstance(instance, Vehicle):
            return instance.start_index
        elif isinstance(instance, Job):
            return instance.location_index

    def calculate_cost(self, matrix, vehicles, routes, vehicle_cost=False, service_cost=True):
        """
        Calculates cost of given route and vehicle pairs
        Args:
            matrix: distance matrix
            vehicles: list of vehicle objects
            routes: list of job objects
            vehicle_cost: if True calculates only for one vehicle, else calculates number of vehicles
            service_cost: if True calculates route cost considering predefined service durations

        Returns:
            Calculated cost value
        """
        cost = 0
        for i in vehicles:
            sub_route = routes[self.get_matrix_index(i)] if not vehicle_cost else routes
            nodes = [i, ]
            [nodes.append(n) for n in sub_route]
            for n in range(1, len(nodes)):
                # print("",nodes[n-1],nodes[n],matrix[nodes[n-1]][nodes[n]])
                c = matrix[self.get_matrix_index(nodes[n - 1])][self.get_matrix_index(nodes[n])]
                # print(c,nodes[n-1],nodes[n])
                s_cost = nodes[n].service if service_cost else 0
                cost = cost + c + s_cost
        return cost

    def is_consistent(self, vehicles, routes):
        """
        Checks route critters have been provided
        Args:
            v:
            r:

        Returns:

        """
        consistency = []
        for v in vehicles:
            route_capacity_need = 0
            sub_route = routes[self.get_matrix_index(v)]
            for r in sub_route:
                route_capacity_need = route_capacity_need + sum(r.delivery)
            consistency.append(True if route_capacity_need <= sum(v.capacity) else False)

        return False if False in consistency else True
