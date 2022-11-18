import click

from ds_metadata_graph_linking.dataset import factory


@click.group()
def cli():
    pass


@cli.command(name='create_complete_hetero_data')
@click.option('--sample_size', type=click.INT, required=True)
@click.option('--raw_data', type=click.STRING, required=True)
@click.option('--raw_graph_data', type=click.STRING, required=True)
@click.option('--processed_data', type=click.STRING, required=True)
@click.option('--num_val', type=click.FLOAT, default=0.1)
@click.option('--num_test', type=click.FLOAT, default=0.1)
@click.option('--neg_sampling_ratio', type=click.FLOAT, default=1)
@click.option('--disjoint_train_ratio', type=click.FLOAT, default=0.1)
@click.option('--add_negative_train_samples/--no-add_negative_train_samples', type=click.BOOL, default=False)
def create_complete_hetero_data(sample_size, raw_data, raw_graph_data, processed_data, num_val, num_test,
                                neg_sampling_ratio, disjoint_train_ratio, add_negative_train_samples):
    factory.create_raw_graph_data_from_raw(sample_size=sample_size,
                                           raw_data=raw_data,
                                           raw_graph_data=raw_graph_data)
    factory.create_hetero_dataset_from_raw_graph_data(raw_graph_data=raw_graph_data,
                                                      processed_data=processed_data)
    factory.split_hetero_data(processed_data=processed_data,
                              num_val=num_val,
                              num_test=num_test,
                              neg_sampling_ratio=neg_sampling_ratio,
                              disjoint_train_ratio=disjoint_train_ratio,
                              add_negative_train_samples=add_negative_train_samples)


@cli.command(name='create_graph_data')
@click.option('--sample_size', type=click.INT, required=True)
@click.option('--raw_data', type=click.STRING, required=True)
@click.option('--raw_graph_data', type=click.STRING, required=True)
def create_raw_graph_data_from_raw(sample_size, raw_data, raw_graph_data):
    factory.create_raw_graph_data_from_raw(sample_size=sample_size,
                                           raw_data=raw_data,
                                           raw_graph_data=raw_graph_data)


@cli.command(name='create_hetero_dataset')
@click.option('--raw_graph_data', type=click.STRING, required=True)
@click.option('--processed_data', type=click.STRING, required=True)
def create_hetero_dataset_from_raw_graph_data(raw_graph_data, processed_data):
    factory.create_hetero_dataset_from_raw_graph_data(raw_graph_data=raw_graph_data,
                                                      processed_data=processed_data)


@cli.command(name='split_hetero_data')
@click.option('--processed_data', type=click.STRING, required=True)
@click.option('--num_val', type=click.FLOAT, default=0.1)
@click.option('--num_test', type=click.FLOAT, default=0.1)
@click.option('--neg_sampling_ratio', type=click.FLOAT, default=1)
@click.option('--disjoint_train_ratio', type=click.FLOAT, default=0.1)
@click.option('--add_negative_train_samples/--no-add_negative_train_samples', type=click.BOOL, default=False)
def train_test_split_hetero_dataset(processed_data, num_val, num_test, neg_sampling_ratio,
                                    disjoint_train_ratio, add_negative_train_samples):
    factory.train_test_split_hetero_dataset(processed_data=processed_data,
                                            num_val=num_val,
                                            num_test=num_test,
                                            neg_sampling_ratio=neg_sampling_ratio,
                                            disjoint_train_ratio=disjoint_train_ratio,
                                            add_negative_train_samples=add_negative_train_samples)


if __name__ == '__main__':
    cli()
