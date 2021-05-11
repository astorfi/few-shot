import os


PATH = os.path.dirname(os.path.realpath(__file__))

PATH = os.path.expanduser('~/exp/few-shot')
DATA_PATH = os.path.expanduser('~/exp/few-shot/data')

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
