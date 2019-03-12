from spiketag.base import bload


def view_data(filename, nCh, fs, dtype, prb=None):
    bf = bload(nCh=nCh, fs=fs)
    bf.load(filename, dtype=dtype)