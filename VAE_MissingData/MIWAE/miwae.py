import torch
import math
import numpy as np
from torch import nn
from torch.distributions import Distribution, Normal, Binomial, MixtureSameFamily, Bernoulli, Categorical, Independent
from ..Utils import safe_log_sum_exp, safe_mean_exp

class MIWAE_VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, prior : Distribution = Normal(0, 1), ):
        super(MIWAE_VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.imputation = lambda x, mask: x*mask # TODO : Add Constant imputation 0
        self.optim_encoder = None
        self.optim_decoder = None
        self.latents_for_eval = self.prior.sample([100, self.encoder.latent_dim])
        self.conditioning_from_eval = None


    def compile(self, optim_encoder, optim_decoder, optim_decoder_mask = None):
        self.optim_encoder = optim_encoder
        self.optim_decoder = optim_decoder



    def forward(self, input, mask = None, iwae_sample_z : int = 1, mc_sample_z : int = 1, return_dict = True):
        n_sample_z = iwae_sample_z * mc_sample_z
        n_sample_x = 1
        batch_size = input.shape[0]
        dim = input.shape[1:]


        if mask is not None :
            current_input = self.imputation(input, mask)
        else :
            current_input = input

        current_input_extended = current_input.unsqueeze(0).unsqueeze(0).expand(iwae_sample_z, mc_sample_z, batch_size, *dim)
        current_input_extended = current_input_extended.flatten(0,2)
        if mask is not None :
            mask_extended = mask.unsqueeze(0).unsqueeze(0).expand(iwae_sample_z, mc_sample_z, batch_size, *dim)
            mask_extended = mask_extended.flatten(0,2)


        _out, _z, q_dist = self.encoder(current_input, iwae_sample_z, mc_sample_z)

        mu, log_var = _out.chunk(2, dim=1)
        
        log_pz = self.prior.log_prob(_z.flatten(0,1))
        log_qz = q_dist.log_prob(_z.flatten(0,1))
        kl = log_qz - log_pz  # shape  [iwae_sample_z * mc_sample_z, batch_size, latent_dim]

        _logits, _output_dist = self.decoder(_z.flatten(0,2), 1) #shape batch_size * iwae_samples * 28 * 28

        log_pgivenz = _output_dist.log_prob(current_input_extended)  
        kl = torch.sum(kl, axis=-1)# sum over the dimension

        if mask is not None:
            log_pgivenz = log_pgivenz * mask_extended
        log_pgivenz = torch.sum(log_pgivenz.reshape(n_sample_z, batch_size, -1), axis=-1) # batch_size * n_samples



        _bound = (log_pgivenz - kl).reshape(iwae_sample_z, mc_sample_z, batch_size)  # batch_size * n_sample
        bound_iwae = torch.logsumexp(_bound, dim = 0) - math.log(iwae_sample_z) # Average over iwae_sample_z
        bound_iwae = torch.mean(bound_iwae, dim = 0) # Average over mc_sample_z
        bound_vae = _bound.mean(axis = [0,1]) # Average over iwae_sample_z and mc_sample_z


        avg_vae_bound = torch.mean(bound_vae)
        avg_iwae_bound = torch.mean(bound_iwae)
        loss = bound_iwae
        _output_dict = {'latents': _z,
                        'q_dist': q_dist,
                        'logits': _logits,
                        'output_dist': _output_dist,
                        'kl': kl,
                        'likelihood': log_pgivenz,
                        'vae_bound': avg_vae_bound,
                        'iwae_bound': avg_iwae_bound
                        }
        if return_dict :
            return loss, _output_dict
        else :
            return loss

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


    def sample_from_input_importance(self, input, mask, n_samples = 10, iwae_sample_z : int = 10000, ):
        mc_sample_z = 1
        batch_size = input.shape[0]
        current_input = self.imputation(input, mask)
        _out, _z, q_dist = self.encoder(current_input, iwae_sample_z, mc_sample_z)
        log_pz = self.prior.log_prob(_z.flatten(0,1))
        log_qz = q_dist.log_prob(_z.flatten(0,1))
        kl = (log_qz - log_pz).flatten(2).sum(axis = -1)  # shape  [iwae_sample_z * mc_sample_z, batch_size, latent_dim]
        _logits, output_dist = self.decoder(_z, 1)
        samples = output_dist.sample()

        log_pgivenz = output_dist.log_prob(samples).reshape(iwae_sample_z, *input.shape).flatten(2).sum(axis = -1) # shape [iwae_sample_z, batch_size, prod(dim)]
        importance_weights = (kl - log_pgivenz)
        importance_weights = torch.exp(importance_weights - safe_log_sum_exp(importance_weights, dim = 0, keepdim = True)).permute(1,0) # Permute batch size and iwae_sample_z
        choices = torch.multinomial(importance_weights, n_samples, replacement = True).permute(0,1) # Permute batch size and n_samples
        samples = samples.reshape(iwae_sample_z, *input.shape) # shape [iwae_sample_z, batch_size, prod(dim)] 

        selected_samples = []
        for k in range(batch_size) :
            selected_samples.append(samples[choices[k, :], k].unsqueeze(1))
        selected_samples = torch.cat(selected_samples, axis = 1)

        return selected_samples

    def sample_from_input(self, input, mask, n_samples = 1,):
        assert n_samples is not None
        # first I have to pass the input through the encoder
        current_input = self.imputation(input, mask)
        _, _z, _ = self.encoder(current_input, mc_sample_z = n_samples)
        # now I have to pass those through the decoder
        _, output_dist = self.decoder(_z, 1)
        
        # now I have to transform these into probabilities
        samples = output_dist.sample()

        return samples

