#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Program entry point"""

from __future__ import print_function

import sys

from deliver.genetic_algorithm.genetic_solver import GeneticSolver
from deliver.problem.problem_helper import ProblemHelper


def main(argv):
    # Read problem JSON from argv
    json_path = sys.argv[1]
    problem = ProblemHelper(json_path)
    # Create Genetic Algorithm solver object
    model = GeneticSolver(problem)
    # Solve the problem
    solution = model.solve()
    solution = model.solve(intermediate_prints=False)
    # Print out the solution
    model.show(solution)
    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


if __name__ == '__main__':
    entry_point()
