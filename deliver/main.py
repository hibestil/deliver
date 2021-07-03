#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Program entry point"""

from __future__ import print_function

import sys

from deliver.genetic_algorithm.genetic_solver import GeneticSolver
from deliver.problem.problem_helper import ProblemHelper


def main(argv):
    generations = 2500
    population_size = 50
    crossover_rate = 0.05
    heuristic_mutate_rate = 0.05
    inversion_mutate_rate = 0.05
    depot_move_mutate_rate = 0
    best_insertion_mutate_rate = 0.1
    route_merge_rate = 0.05

    json_path = sys.argv[1]
    problem = ProblemHelper(json_path)
    model = GeneticSolver(problem,population_size=population_size)
    solution = model.solve(generations, crossover_rate, heuristic_mutate_rate, inversion_mutate_rate,
                           depot_move_mutate_rate, best_insertion_mutate_rate, route_merge_rate)

    model.show(solution)
    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


if __name__ == '__main__':
    entry_point()
