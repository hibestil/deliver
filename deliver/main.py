#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Program entry point"""

from __future__ import print_function

import sys

from deliver.dataset_helper import DatasetHelper


def main(argv):
    json_path = sys.argv[1]
    dataset_helper = DatasetHelper(json_path)
    vehicles, jobs, matrix = dataset_helper.process_data()
    [print(v) for v in vehicles]
    [print(j) for j in jobs]
    print(matrix)
    return 0


def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))


if __name__ == '__main__':
    entry_point()
