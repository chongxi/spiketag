import spiketag
import click



@click.group()
def main():
    pass

@main.command()
@click.argument('binaryfile')
@click.argument('probefile')
@click.option('--nCh', prompt='nCh', default='160')
@click.option('--fs', prompt='fs', default='25000.0')
@click.option('--bits', prompt='bits', default='16')
@click.option('--time', prompt='time', default='0')
@click.option('--span', prompt='span', default='5')
def check(binaryfile, probefile, nCh, bits, time, span):
    click.echo('spiketag-check {}, {}bits, time={}, span={}'.format(file, bits, time, span))


@main.command()
def sort():
    click.echo('spiketag-sort')

@main.command()
def load():
    click.echo('spiketag-load')    


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
