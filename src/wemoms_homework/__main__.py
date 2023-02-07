import click
from wemoms_homework.data.make_dataset import dataset
from wemoms_homework.features.build_features import build
from wemoms_homework.features.merge_features import merge
from wemoms_homework.models.train_model import train
from wemoms_homework.models.predict_model import predict

cli = click.CommandCollection(sources=[
    dataset,
    build,
    merge,
    predict,
    train,
])

if __name__ == '__main__':
    cli()
