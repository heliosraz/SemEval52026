#!/usr/bin/env python3
from argparse import ArgumentParser

def runtime_arguments():
    parser = ArgumentParser()
    g = parser.add_argument_group("-o", "--ollama")
    g.add_argument("-m", "--model",
                   required=True)
    g.add_argument("-c", "--cloud",
                   help="run model online",
                   default=False)

    return parser.parse_args()
