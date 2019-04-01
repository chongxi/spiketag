import os
import os.path as op
import shutil
import spiketag


def root_dir():
    home = op.realpath(op.expanduser('~'))
    return op.join(home, '.spiketag')

def config_dir():
    home = op.realpath(op.expanduser('~'))
    return op.join(home, '.spiketag', 'spiketag')

def origin_dir():
    return op.join(spiketag.__path__[0], 'res', 'spiketag')

def res_dir():
    return op.join(spiketag.__path__[0], 'res')

def gen_config():
    config_file = op.join(config_dir(), 'state.json') 
    origin_file = op.join(origin_dir(), 'state.json')
    if not op.exists(config_file):
        os.makedirs(config_dir())
        shutil.copy(op.realpath(origin_file), config_dir())

gen_config() # gen state.json file in user's home directory if doesn't exists

from .Control import controller
from .Model import MainModel
from .View import MainView
