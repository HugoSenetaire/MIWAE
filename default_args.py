import argparse
from MissingDataDataset.default_args import default_args_missingdatadataset 
from VAE_MissingData.default_args import default_args_miwae
import yaml


def open_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def update_config_from_paths(args_dict,):


    if args_dict["yamldataset"] is not None :
        args_dict.update(open_yaml(args_dict["yamldataset"]))

    if args_dict["yamlmodel"] is not None :
        args_dict.update(open_yaml(args_dict["yamlmodel"]))


    return args_dict


def default_args(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser = default_args_missingdatadataset(parser = parser, root_default="./local/")
    parser = default_args_miwae(parser = parser)


    args = parser.parse_args()
    args_dict = vars(args)
    args_dict = update_config_from_paths(args_dict)

    return args_dict