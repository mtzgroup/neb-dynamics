"""
this module contains helper general functions
"""
import cProfile
import json
import multiprocessing as mp
import pickle
import signal
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.core.display import HTML


def print_matrix_2D(mat: np.array, precision: float = 6, threshold: float = 0.0):
    """
    given a numpy 2d array, returns the pandas matrix (making it beautiful in jupyter)
    """
    pd.set_option("precision", precision)
    pd.set_option("chop_threshold", threshold)
    (siza, _) = mat.shape
    indexes = np.arange(siza) + 1
    out = pd.DataFrame(mat, index=indexes, columns=indexes)
    return out


def pairwise(iterable):
    """
    from a list [a,b,c,d] to [(a,b),(b,c),(c,d)]
    """
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)
        a = b


def read_json(fn: Path):
    with open(fn, "r") as f:
        data = json.load(f)
    return data


def psave(thing, fn):
    """this quickle saves an object to disc"""
    pickle.dump(thing, open(fn, "wb"))


def pload(fn):
    """this quickle loads an object from disc"""
    return pickle.load(open(fn, "rb"))


def draw_starting_node(tree, name, size=(500, 500)):
    """
    This function is used to make presentations
    """
    molecule = tree.nodes[0]["molecule_list"][0]
    string = f"""<h2>{name.replace('_',' ').capitalize()}</h2><h3>Smiles: {molecule.smiles}</h3>
    <div style="width: 100%; display: table;"> <div style="display: table-row;">
    <div style="width: 30%; display: table-cell;">
    <p style="text-align: left;"><b>Rdkit visualization</b></p>
    {molecule.draw(mode='rdkit', string_mode=True)}
    </div>
    <div style="width: 10%; display: table-cell; vertical-align: middle;">
    <b>---></b>
    </div>
    <div style="width: 50%; display: table-cell;">
    <p style="text-align: left;"><b>Internal Graph visualization</b></p>
    {molecule.draw(mode='d3', string_mode=True, size=size, node_index=False, percentage=0.5)}
    </div>
    </div></div>"""
    return HTML(string)


def load_pickle(fn):
    """
    tedious to remember protocol flag and stuffs
    fn :: FilePath
    """
    return pickle.load(open(fn, "rb"))


def save_pickle(thing, fn):
    """
    tedious part 2
    fn :: FilePath
    thing :: Structure to save
    """
    with open(fn, "wb") as pickle_file:
        pickle.dump(thing, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def wrapper(fn, tup1, tup2):
    fn(*tup1, *tup2)


def execute_parallel(function, iterator, repeated_arguments):
    with mp.Pool() as pool:
        pool.starmap(wrapper, zip(repeat(function), iterator, repeat(repeated_arguments)))


def execute_serial(function, iterator, repeated_arguments):
    for iter in iterator:
        print(f"\n\nI am executing {function} in {iter} with {len(repeated_arguments)} repeated arguments")
        function(*iter, *repeated_arguments)


def profile_this_function(func):
    """
    a decorator for activating the profiler
    """

    def inner1(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        func(*args, **kwargs)
        pr.disable()
        pr.print_stats()

    return inner1
