from spiketag.base import probe
from spiketag.base import MUA
from spiketag.fpga import FPGA
from vispy import app


def view_data(filename, prbfile, nbits, time=0, span=5):
    prb = probe()
    prb.load(prbfile)
    mua = MUA(filename, prb, numbytes=nbits//8, scale=False)
    if span == -1: span=None
    mua.show(chs=prb.chs[:128], span=span, time=time)
    app.run()


def check_fpga(prbfile, var):
    prb = probe()
    prb.load(prbfile)    
    prb.n_ch = 160
    prb.fs  = 25000.
    fpga = FPGA(prb)
    if var in dir(fpga):
        exec('print(fpga.{})'.format(var))

