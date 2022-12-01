import torch
from .Networks import dic_network
from .ReparametrizationTrick import dic_reparametrization
from .decoder import Decoder
from .encoder import Encoder
from .MIWAE import MIWAE_VAE, NotMIWAE_VAE
from torch.distributions import Normal
dic_prior = {
    "normal01" : Normal(0,1),
}


def get_networks(args_dict, input_channel =1):
    encoder_network = dic_network[args_dict["encoder_network"]]
    decoder_network = dic_network[args_dict["decoder_network"]]
    if args_dict["decoder_mask_network"] is not None:
        decoder_mask_network = dic_network[args_dict["decoder_mask_network"]]
    else :
        decoder_mask_network = None
    return encoder_network, decoder_network, decoder_mask_network

def get_encoder(args_dict, encoder_network):
    reparam_encoder = dic_reparametrization[args_dict["encoder_reparam"]]()
    input_size = args_dict["input_size"]
    multiplyer_for_param = reparam_encoder.multiplyer_for_param
    latent_dim = args_dict["latent_dim"]
    encoder_network = encoder_network(input_size = input_size, latent_dim = latent_dim * multiplyer_for_param)
    encoder = Encoder(reparam_trick=reparam_encoder, encoder_network=encoder_network, )
    return encoder

def get_decoder(args_dict, decoder_network = None, ):
    reparam_decoder = dic_reparametrization[args_dict["decoder_reparam"]]()
    decoder_network = decoder_network(input_size = (1, args_dict['latent_dim']), output_size = args_dict['input_size'])
    decoder = Decoder(reparam_trick=reparam_decoder, decoder_network=decoder_network,)
    return decoder

def get_decoder_mask(args_dict, decoder_mask_network = None, output_channel = 1,):
    if (decoder_mask_network is not None) and (args_dict["decoder_mask_network"] is not None):
        reparam_decoder_mask = dic_reparametrization[args_dict["decoder_mask_reparam"]]()
        output_size = (1, *args_dict['input_size'][1:])
        input_size = args_dict['input_size']
        decoder_mask_network = decoder_mask_network(input_size = input_size, output_size = output_size)
        decoder_mask = Decoder(reparam_trick=reparam_decoder_mask, decoder_network = decoder_mask_network, )
    else :
        assert args_dict["model_masking_process"] is None, "Gave no options for parametrization of the decoder mask but want to model masking process"
        decoder_mask = None
    return decoder_mask



def get_miwae(args_dict, encoder, decoder, decoder_mask = None):
    prior = dic_prior[args_dict["prior"]]
    if not args_dict["model_masking_process"]:
        miwae = MIWAE_VAE(encoder, decoder, prior = prior,)
    else :
        miwae = NotMIWAE_VAE(encoder, decoder, decoder_mask, prior = prior,)
    return miwae
