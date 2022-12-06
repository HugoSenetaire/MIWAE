import argparse
from .Networks import dic_network
from .ReparametrizationTrick import dic_reparametrization
from .get_miwae import dic_prior



def default_args_miwae(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--model_masking_process", action = 'store_true', default = False)

    parser.add_argument("--encoder_network", type = str, choices=dic_network.keys(), default = "ConvFeatureExtractor")
    parser.add_argument("--decoder_network", type = str, choices= dic_network.keys(), default = "ConvDecoder")
    parser.add_argument("--decoder_mask_network", type = str, choices= dic_network.keys(), default = None)

    parser.add_argument("--prior", type = str, default= "normal01", choices=dic_prior.keys())

    parser.add_argument("--encoder_reparam", type = str, choices=dic_reparametrization.keys(), default= "ReparamTrickNormal")
    parser.add_argument("--decoder_reparam", type = str, choices=dic_reparametrization.keys(), default= "ReparamTrickBernoulli")
    parser.add_argument("--decoder_mask_reparam", type = str, choices=dic_reparametrization.keys(), default = None)

    parser.add_argument("--iwae_z_test", type = int, default= 1)
    parser.add_argument("--iwae_z", type = int, default=1)
    parser.add_argument("--mc_z_test", type = int, default=1)
    parser.add_argument("--mc_z", type = int, default=1)

    parser.add_argument("--root_dir", type = str, default="./local/")
    parser.add_argument("--latent_dim", type = int, default = 10)

    parser.add_argument("--use_wandb", action="store_true")

    parser.add_argument("--lr_encoder", type = float, default = 1e-3)
    parser.add_argument("--lr_decoder", type = float, default = 1e-3)
    parser.add_argument("--lr_decoder_mask", type = float, default = 1e-3)
    parser.add_argument("--constant_imputation", type = int, default = 0)

    
    parser.add_argument('--nb_epoch', type=int, default=100)
    

    parser.add_argument('--yamlmodel', type=str, default=None)

    return parser

    