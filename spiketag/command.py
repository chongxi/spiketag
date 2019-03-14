import click


@click.group()
def main():
    pass


@main.command()
@click.argument('binaryfile')
@click.argument('probefile')
@click.option('--nbits', prompt='nbits', default='32')
@click.option('--time', prompt='time', default='0')
@click.option('--span', prompt='span', default='10')
def check(binaryfile, probefile, nbits, time, span):
    from .spiketag import view_data
    view_data(binaryfile, probefile, int(nbits), float(time), float(span))


@main.command()
def sort():
    click.echo('spiketag-sort')


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
    # if var in dir(fpga):
    #     exec('print(fpga.{})'.format(var))  


@main.command()
@click.argument('probefile')
def fpga(probefile):
    # from spiketag import check_fpga
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
