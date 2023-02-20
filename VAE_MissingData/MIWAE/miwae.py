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
        self.optim_decoder_mask = optim_decoder_mask

    def forward(self, input, mask,  iwae_sample_z : int=1, mc_sample_z: int=1, sample_pathwise = None, return_dict= True):
        """
        Calculate the loss for the MIWAE VAE as well as the KL divergence and the log likelihood
        We consider that everything is already extended in here and we want to use the samples previously created
        for the reparam trick. The goal is to use vmap.
        """
        batch_size = input.shape[0]
        dim = input.shape[1:]
        mask_dim = mask.shape[1:]



        # Calculate the KL divergence
        current_input = self.imputation(input, mask)
        current_input_extended = current_input.unsqueeze(1).unsqueeze(1).expand(batch_size, iwae_sample_z, mc_sample_z, *dim)
        mask_extended = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, iwae_sample_z, mc_sample_z, *mask_dim)


        param_z = self.encoder.encoder_network(current_input,)
        param_z = param_z.unsqueeze(1).unsqueeze(1).expand(batch_size, iwae_sample_z, mc_sample_z, -1)
        if sample_pathwise is None:
            sample_pathwise = self.encoder.reparam_trick.sample_pathwise((batch_size, iwae_sample_z, mc_sample_z, self.encoder.latent_dim)).to(param_z.device)
        sample_pathwise = sample_pathwise.reshape(batch_size, iwae_sample_z, mc_sample_z, self.encoder.latent_dim)

        _z = self.encoder.reparam_trick.rsample(param_z, sample_pathwise)
        log_pz = -0.5 * torch.log(torch.tensor(2*torch.pi, device=_z.device)) - (_z**2)/2
        log_qz = self.encoder.reparam_trick.log_prob(_z, param_z,)
        kl = (log_qz - log_pz).reshape(batch_size, iwae_sample_z, mc_sample_z, -1).sum(-1)

        # Calculate the log likelihood
        param_x = self.decoder.decoder_network(_z.reshape(batch_size*iwae_sample_z*mc_sample_z, -1)).reshape(batch_size, iwae_sample_z, mc_sample_z, -1)
        log_pgivenz = self.decoder.reparam_trick.log_prob(current_input_extended.reshape(batch_size, iwae_sample_z, mc_sample_z, -1), param_x,)
        log_pgivenz = log_pgivenz.reshape(batch_size, iwae_sample_z, mc_sample_z, -1)*mask_extended.reshape(batch_size, iwae_sample_z, mc_sample_z, -1)
        log_pgivenz = log_pgivenz.sum(-1)



        likelihood_estimated = (log_pgivenz.reshape(batch_size, iwae_sample_z, mc_sample_z,).logsumexp(1) -math.log(iwae_sample_z)).mean(1)
        kl_estimated = (kl.reshape(batch_size, iwae_sample_z, mc_sample_z,).logsumexp(1) - math.log(iwae_sample_z)).mean(1)
        log_pz_estimated = (log_pz.reshape(batch_size, iwae_sample_z, mc_sample_z,-1).sum(-1).logsumexp(1) - math.log(iwae_sample_z)).mean(1)
        log_qz_estimated = (log_qz.reshape(batch_size, iwae_sample_z, mc_sample_z,-1).sum(-1).logsumexp(1) - math.log(iwae_sample_z)).mean(1)

        # Calculate the bounds
        _bound = (log_pgivenz - kl).reshape(batch_size, iwae_sample_z, mc_sample_z,) 
        bound_iwae = torch.logsumexp(_bound, dim = 1) - math.log(iwae_sample_z) # Average over iwae_sample_z
        bound_iwae = torch.mean(bound_iwae, dim = 1) # Average over mc_sample_z
        bound_vae = _bound.mean(axis = [1,2]) # Average over iwae_sample_z and mc_sample_z
        loss = bound_iwae


        _output_dict = {'latents': _z.reshape(batch_size, iwae_sample_z, mc_sample_z, -1),
                        'log_pz': log_pz_estimated,
                        'log_qz': log_qz_estimated,
                        'kl': kl_estimated,
                        'likelihood': likelihood_estimated,
                        'vae_bound': bound_vae,
                        'iwae_bound': bound_iwae,
                        }
        return loss, _output_dict




    def forward_original(self, input, mask = None, pathwise_sample = None, iwae_sample_z : int = 1, mc_sample_z : int = 1, return_dict = True):
        batch_size = input.shape[0]
        dim = input.shape[1:]
        if mask is not None :
            current_input = self.imputation(input, mask)
        else :
            current_input = input

        current_input_extended = current_input.unsqueeze(1).unsqueeze(1).expand(batch_size, iwae_sample_z, mc_sample_z, *dim)
        current_input_extended = current_input_extended.flatten(0,2)
        if mask is not None :
            mask_extended = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, iwae_sample_z, mc_sample_z, *dim)
            mask_extended = mask_extended.flatten(0,2)


        # Calculate the KL divergence
        # print(_z.shape)

            # _out = self.encoder.encoder_network(current_input_extended)
        _out, _z_init, q_dist_expanded = self.encoder(current_input, iwae_sample_z = iwae_sample_z, mc_sample_z = mc_sample_z)
        _out_expanded = _out.unsqueeze(1).unsqueeze(1).expand(batch_size, iwae_sample_z, mc_sample_z, -1)
        if pathwise_sample is not None :
            _z = self.encoder.reparam_trick.rsample(_out_expanded, pathwise_sample.reshape(batch_size, iwae_sample_z, mc_sample_z, -1))
        else :
            _z = _z_init



        log_pz = self.prior.log_prob(_z)
        log_qz = self.encoder.reparam_trick.log_prob(_z, _out_expanded)
        if pathwise_sample is not None :
            log_qz_2 = self.encoder.reparam_trick.log_prob(_z, _out_expanded)
        kl = (log_qz - log_pz).reshape(batch_size, iwae_sample_z, mc_sample_z, -1)  # shape  [iwae_sample_z * mc_sample_z, batch_size, latent_dim]
        kl = torch.sum(kl, axis=-1)# sum over the dimension


        _logits, _output_dist = self.decoder(_z.flatten(0,2), 1) #shape batch_size * iwae_samples * 28 * 28
        log_pgivenz = _output_dist.log_prob(current_input_extended).reshape(batch_size, iwae_sample_z, mc_sample_z, np.prod(dim))
        if mask is not None:
            log_pgivenz = log_pgivenz * mask_extended.reshape(batch_size, iwae_sample_z, mc_sample_z, np.prod(dim))
        log_pgivenz = torch.sum(log_pgivenz.reshape(batch_size, iwae_sample_z, mc_sample_z, -1), axis=-1) # batch_size * n_samples



        _bound = (log_pgivenz - kl).reshape(batch_size, iwae_sample_z, mc_sample_z,)  # batch_size * n_sample
        bound_iwae = torch.logsumexp(_bound, dim = 1) - math.log(iwae_sample_z) # Average over iwae_sample_z
        bound_iwae = torch.mean(bound_iwae, dim = 1) # Average over mc_sample_z
        bound_vae = _bound.mean(axis = [1,2]) # Average over iwae_sample_z and mc_sample_z
        loss = bound_iwae

        log_pz_estimated = (log_pz.reshape(batch_size, iwae_sample_z, mc_sample_z,-1).sum(-1).logsumexp(1) - math.log(iwae_sample_z)).mean(1)
        log_qz_estimated = (log_qz.reshape(batch_size, iwae_sample_z, mc_sample_z,-1).sum(-1).logsumexp(1) - math.log(iwae_sample_z)).mean(1)
        kl_estimated = (kl.reshape(batch_size, iwae_sample_z, mc_sample_z,).logsumexp(1) - math.log(iwae_sample_z)).mean(1)
        likelihood_estimated = (log_pgivenz.reshape(batch_size, iwae_sample_z, mc_sample_z,).logsumexp(1) - math.log(iwae_sample_z)).mean(1)

        _output_dict = {'latents': _z.reshape(batch_size, iwae_sample_z, mc_sample_z, -1),
                        'log_pz': log_pz_estimated,
                        'log_qz': log_qz_estimated,
                        'logits': _logits,
                        'output_dist': _output_dist,
                        'kl': kl_estimated,
                        'likelihood': likelihood_estimated,
                        'vae_bound': bound_vae,
                        'iwae_bound': bound_iwae,
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
        log_qz = self.encoder.dist.log_prob(_z.flatten(0,1))
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

