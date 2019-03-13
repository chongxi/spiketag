from spiketag.base import MUA
from spiketag.base import probe
from vispy import app


def view_data(filename, prbfile, nCh, fs, nbits, time=0, span=5):
    prb = probe()
    prb.load(prbfile)
    prb.n_ch = nCh
    prb.fs  = fs
    mua = MUA(filename, prb, numbytes=nbits//8, scale=False)
    mua.show(chs=prb.chs[:128], span=span, time=time)
    app.run()
    