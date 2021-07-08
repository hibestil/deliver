#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Program entry point"""

from __future__ import print_function

import argparse
import json
import sys

from deliver.genetic_algorithm.brute_force_solver import BruteForceSolver
from deliver.genetic_algorithm.genetic_solver import GeneticSolver
from deliver.problem.problem_helper import ProblemHelper


def main(argv):
    parser = argparse.ArgumentParser(description='Deliver MDVRP | Genetic algoritms for solving MDVRP problems')
    parser.add_argument("-i", dest="input_filename", required=True,
                        help="input file with json format.", metavar="FILE",
                        type=str)
    parser.add_argument('--benchmark_input', dest="benchmark_input_type", default=False, action='store_true')
    parser.add_argument('--brute_force', dest="brute_force", default=False, action='store_true')
    parser.add_argument('--intermediate_prints', dest="intermediate_prints", default=False, action='store_true')
    parser.add_argument("-o", dest="output_filename", required=True,
                        help="output file path.", metavar="FILE", type=str)
    args = parser.parse_args()

    # Read problem JSON from argv
    problem = ProblemHelper(args.input_filename, benchmark=args.benchmark_input_type)
    # Create Genetic Algorithm solver object
    if args.brute_force:
        model = BruteForceSolver(problem)
    else:
        model = GeneticSolver(problem)

    # Solve the problem
    print("Solving process has been started...")
    solution = model.solve(intermediate_prints=args.benchmark_input_type)
    # Print out the solution
    json_data = model.create_output_json(solution)
    # Generate output
    out_file = open(args.output_filename, "w")
    json.dump(json_data, out_file, indent=6)
    out_file.close()
    print("Output created:", args.output_filename)
    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


if __name__ == '__main__':
    entry_point()
