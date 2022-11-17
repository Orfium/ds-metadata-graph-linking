import click

from ds_metadata_graph_linking.dataset.factory import create_graph_dataset_from_raw


@click.group()
def cli():
    pass


@cli.command(name='create_dataset')
@click.option('--sample_size', type=click.INT, required=True)
@click.option('--raw_data', type=click.STRING, required=True)
@click.option('--dataset_to_save', type=click.STRING, required=True)
def create_dataset_from_raw(sample_size, raw_data, dataset_to_save):
    create_graph_dataset_from_raw(sample_size=sample_size, raw_data=raw_data, dataset_to_save=dataset_to_save)


if __name__ == '__main__':
    cli()

