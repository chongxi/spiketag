import spiketag
import click



@click.command()
def main():
    click.echo('click.main()')
    return 0



if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover