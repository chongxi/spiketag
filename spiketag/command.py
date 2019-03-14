import click


@click.group()
def main():
    pass


@main.command()
@click.argument('binaryfile', nargs=-1)
@click.argument('probefile')
@click.option('--nbits', prompt='nbits', default='32')
@click.option('--chs', prompt='chs', default='0,128')
@click.option('--time', prompt='time', default='0')
@click.option('--span', prompt='span', default='10')
def view(binaryfile, probefile, nbits, chs, time, span):
    from spiketag.base import probe
    from spiketag.base import MUA 
    from vispy import app
    nbits, time, span = int(nbits), float(time), float(span)
    chs_tbview = slice(*[int(_) for _ in chs.split(',')])
    prb = probe()
    prb.load(probefile)
    if len(binaryfile) == 2:
        mua_filename, spk_filename = binaryfile
        click.echo('loadding {} and {}'.format(mua_filename, spk_filename))
        mua = MUA(mua_filename=mua_filename, spk_filename=spk_filename,
                  probe=prb, numbytes=nbits//8, scale=False)
        if span == -1: span=None
        mua.show(chs=prb.chs[chs_tbview], span=span, time=time)
        app.run()
    elif len(binaryfile) == 1:
        mua_filename = binaryfile[0]
        click.echo('loadding {}'.format(mua_filename))
        mua = MUA(mua_filename=mua_filename, spk_filename=None,
                  probe=prb, numbytes=nbits//8, scale=False)
        if span == -1: span=None
        mua.show(chs=prb.chs[chs_tbview], span=span, time=time)
        app.run()


# @click.option('--time_cutoff', prompt='time_cutoff', default='0')
@main.command()
@click.argument('binaryfile', nargs=-1)
@click.argument('probefile')
def sort(binaryfile, probefile):
    mua_filename, spk_filename = binaryfile
    click.echo('spiketag-sort: loadding {} and {}'.format(mua_filename, spk_filename))
    import sys
    from PyQt5.QtWidgets import QApplication
    from spiketag.mvc.Control import controller
    from spiketag.base import probe
    prb = probe()
    prb.load(probefile)
    app  = QApplication(sys.argv)
    ctrl = controller(
                      probe = prb,
                      mua_filename=mua_filename, 
                      spk_filename=spk_filename, 
                      binary_radix=13, 
                      scale=False
                      # time_segs=[[0,320]]
                     )
    ctrl.model.sort(clu_method='dpgmm', group_id=0, n_comp=8, max_iter=400)
    ctrl.show()
    sys.exit(app.exec_())


@main.command()
@click.option('--var', prompt='ch_grpNo, ch_hash, ch_ref, thres, scale, shift, pca, vq')
def fpga_check(var):
    click.echo('check FPGA')
    from spiketag.fpga import xike_config
    fpga = xike_config()
    if var in dir(fpga):
        exec('print(fpga.{})'.format(var))  


@main.command()
@click.argument('cmd')
def fpga_detector(cmd):
    click.echo('set FPGA detector {} '.format(cmd))
    from spiketag.fpga import xike_config
    fpga = xike_config()
    exec('fpga.{}'.format(cmd))  


@main.command()
@click.argument('probefile')
def fpga(probefile):
    click.echo('init FPGA with probe file {}'.format(probefile))
    from spiketag.fpga import xike_config
    from spiketag.base import probe
    prb = probe()
    prb.load(probefile)    
    prb.show()
    fpga = xike_config(prb)
    click.echo('init FPGA success, check fpga with "spiketag fpga-check"')


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
