#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Program entry point"""

from __future__ import print_function

import sys

from deliver.genetic_algorithm.genetic_solver import GeneticSolver
from deliver.problem.problem_helper import ProblemHelper


def main(argv):
    # Genetic Algorithm Parameters
    generations = 2500
    population_size = 50
    crossover_rate = 0.05
    heuristic_mutate_rate = 0.05
    inversion_mutate_rate = 0.05
    depot_move_mutate_rate = 0
    best_insertion_mutate_rate = 0.1
    route_merge_rate = 0.05
    # Read problem JSON from argv
    json_path = sys.argv[1]
    problem = ProblemHelper(json_path)
    # Create Genetic Algorithm solver object
    model = GeneticSolver(problem, population_size=population_size)
    # Solve the problem
    solution = model.solve(generations, crossover_rate, heuristic_mutate_rate, inversion_mutate_rate,
                           depot_move_mutate_rate, best_insertion_mutate_rate, route_merge_rate)
    # Print out the solution
    model.show(solution)
    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


if __name__ == '__main__':
    entry_point()
