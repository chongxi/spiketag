import spiketag
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
    click.echo('spiketag-check {}:{} channels, {}bits, time={}, span={}'.format(binaryfile, nch, nbits, time, span))
    spiketag.view_data(binaryfile, probefile, int(nch), float(fs), int(nbits), float(time), float(span))


@main.command()
def sort():
    click.echo('spiketag-sort')

@main.command()
def load():
    click.echo('spiketag-load')    


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
