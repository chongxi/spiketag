import click

@click.group()
def main():
    pass

@main.command()
@click.argument('binaryfile')
@click.argument('probefile')
@click.option('--nch', prompt='nch', default='160')
@click.option('--fs', prompt='fs', default='25000.0')
@click.option('--nbits', prompt='nbits', default='16')
@click.option('--time', prompt='time', default='0')
@click.option('--span', prompt='span', default='5')
def check(binaryfile, probefile, nch, fs, nbits, time, span):
    # from spiketag import view_data
    click.echo('spiketag-check {}:{} channels, {}bits, time={}, span={}'.format(binaryfile, nch, nbits, time, span))
    # view_data(binaryfile, probefile, int(nch), float(fs), int(nbits), float(time), float(span))

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
