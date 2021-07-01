from deliver.problem_solver import ProblemSolver


class GeneticSolver(ProblemSolver):
    groups = None

    def __init__(self, problem):
        super().__init__(problem)
        self.group_customers()

    def solve(self):
        return

    def group_customers(self):
        self.groups = [[] for i in range(len(self.problem.depots))]
        # Group customers to closest depot
        for c in self.problem.customers:
            depot, depot_index, dist = self.find_closest_depot(c)
            self.groups[depot_index].append(c)
        print(self.groups)

    def find_closest_depot(self, customer_id):
        closest_depot = None
        closest_distance = -1
        for i, depot in enumerate(self.problem.depots):
            from_c_to_d = self.problem.matrix.data[i][customer_id]
            from_d_to_c = self.problem.matrix.data[customer_id][i]
            total_distance = from_c_to_d + from_d_to_c
            if closest_depot is None or total_distance < closest_distance:
                closest_depot = (depot, i)
                closest_distance = total_distance

        return closest_depot[0], closest_depot[1], closest_distance
