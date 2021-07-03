#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Program entry point"""

from __future__ import print_function

import json
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
    solution = model.solve(intermediate_prints=False)
    # Print out the solution
    model.show(solution)
    json_data = model.create_output_json(solution)

    # the json file where the output must be stored
    out_file = open(r"../data/output.json", "w")
    json.dump(json_data, out_file, indent=6)
    out_file.close()

    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


if __name__ == '__main__':
    entry_point()
