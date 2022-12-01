import torch
import math
from torch import nn
from torch.distributions import Distribution, Normal, Binomial, MixtureSameFamily, Bernoulli, Categorical, Independent

class NotMIWAE_VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, decoder_mask: nn.Module, prior : Distribution = Normal(0, 1), ):
        super(NotMIWAE_VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.decoder_mask = decoder_mask
        self.prior = prior
        self.imputation = lambda x, mask: x*mask # TODO : Add Constant imputation 0
        self.optim_encoder = None
        self.optim_decoder = None
        self.optim_decoder_mask = None
        self.latents_for_eval = self.prior.sample([100, self.encoder.latent_dim])
        self.conditioning_from_eval = None


    def compile(self, optim_encoder, optim_decoder, optim_decoder_mask):
        self.optim_encoder = optim_encoder
        self.optim_decoder = optim_decoder
        self.optim_decoder_mask = optim_decoder_mask



    def forward(self, input, mask = None, iwae_sample_z : int = 1, mc_sample_z : int = 1, ):
        n_sample_z = iwae_sample_z * mc_sample_z
        n_sample_x = 1
        batch_size = input.shape[0]
        dim = input.shape[1:]

        assert mask is not None, "Mask is not provided for notMIWAE"

        current_input = self.imputation(input, mask)

        current_input_extended = current_input.unsqueeze(0).unsqueeze(0).expand(iwae_sample_z, mc_sample_z, batch_size, *dim)
        current_input_extended = current_input_extended.flatten(0,2)
        mask_extended = mask.unsqueeze(0).unsqueeze(0).expand(iwae_sample_z, mc_sample_z, batch_size, *dim)
        mask_extended = mask_extended.flatten(0,2)


        _out, _z, q_dist = self.encoder(current_input, iwae_sample_z, mc_sample_z)
        
        log_pz = self.prior.log_prob(_z.flatten(0,1))
        log_qz = q_dist.log_prob(_z.flatten(0,1))
        kl = log_qz - log_pz  # shape  [iwae_sample_z * mc_sample_z, batch_size, latent_dim]

        _logits, _output_dist = self.decoder(_z.flatten(0,2), 1) #shape batch_size * iwae_samples * 28 * 28

        log_pxgivenz = _output_dist.log_prob(current_input_extended) 
        
        
        x_sampled = _output_dist.sample() #shape iwae samples * mc samples, batch_size, 28, 28
        # print(x_sampled.shape)
        # print(current_input_extended.shape)
        # print(mask_extended.shape)
        x_mix = x_sampled * (1-mask_extended) + current_input_extended * mask_extended

        # log_pmaskgivenx = _output_dist.log_prob(mask_extended)
        # print(x_mix.shape)
        _logits_mask, _output_dist_mask = self.decoder_mask(x_mix, 1) #shape batch_size * iwae_samples * 28 * 28
        log_pmaskgivenx = _output_dist_mask.log_prob(mask_extended)

        likelihood = log_pxgivenz + log_pmaskgivenx
        likelihood = likelihood * mask_extended
        kl = torch.sum(kl, axis=-1)# sum over the dimension
        likelihood = torch.sum(likelihood.reshape(n_sample_z, batch_size, -1), axis=-1) # batch_size * n_samples

        _bound = (likelihood - kl).reshape(iwae_sample_z, mc_sample_z, batch_size)  # batch_size * n_sample
        bound_iwae = torch.logsumexp(_bound, dim = 0) - math.log(iwae_sample_z) # Average over iwae_sample_z
        bound_iwae = torch.mean(bound_iwae, dim = 1) # Average over mc_sample_z
        bound_vae = _bound.mean(axis = [0,1]) # Average over iwae_sample_z and mc_sample_z


        avg_vae_bound = torch.mean(bound_vae)
        avg_iwae_bound = torch.mean(bound_iwae)

        _output_dict = {'latents': _z,
                        'q_dist': q_dist,
                        'logits': _logits,
                        'output_dist': _output_dist,
                        'kl': kl,
                        'likelihood': likelihood,
                        'vae_bound': avg_vae_bound,
                        'iwae_bound': avg_iwae_bound
                        }
        
        return _output_dict


    def sample_from_prior(self, n_samples = None, latents = None):
        assert n_samples is not None or latents is not None
        if latents is None:
            latents = self.prior.sample([n_samples, self.encoder.latent_dim])
        # print(latents.shape)
        # now I have to pass those through the decoder
        _logits, output_dist = self.decoder(latents, None)
        
        # now I have to transform these into probabilities
        samples = output_dist.sample()

        return _logits, samples


    def sample_from_input(self, input, mask, n_samples = 1,):
        assert n_samples is not None
        # first I have to pass the input through the encoder
        _, _z, _ = self.encoder(input, mc_sample_z = n_samples)
        # now I have to pass those through the decoder
        _, output_dist = self.decoder(_z, 1)
        
        # now I have to transform these into probabilities
        samples = output_dist.sample()

        return samples

