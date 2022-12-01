import click

from data_fetcher.generate_nodes import generate_nodes


@click.group()
def cli():
    pass


@cli.command(name='generate_raw_data')
@click.option('--asset_full_path', type=click.STRING, required=True)
@click.option('--asset_share_path', type=click.STRING, required=True)
@click.option('--raw_data_path', type=click.STRING, required=True)
def generate_raw_data(asset_full_path, asset_share_path, raw_data_path):
    generate_nodes(asset_full_path=asset_full_path, asset_share_path=asset_share_path, raw_data_path=raw_data_path)


if __name__ == '__main__':
    cli()
