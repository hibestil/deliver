#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Program entry point"""

from __future__ import print_function

import sys

from deliver.genetic_solver import GeneticSolver
from deliver.problem_helper import ProblemHelper
from deliver.solve import Solve


def main(argv):
    json_path = sys.argv[1]
    problem = ProblemHelper(json_path)
    model = GeneticSolver(problem)
    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


if __name__ == '__main__':
    entry_point()
