
from ..trainer_step_default import TrainerStepDefault
from torch.autograd import grad
from functools import partial
from functorch import make_functional_with_buffers, vmap, grad


import logging
import torch

class TrainerStepFunctorchGradNorm(TrainerStepDefault):
    def __init__(self, onepass,  optim_list, scheduler_list = []) -> None:
        super().__init__(onepass,
                        optim_list,
                        scheduler_list = scheduler_list,
                        )


        

    # def single_gradient_test(self,grad_weight_to_compare, data_expanded, mask_expanded, pathwise_sample, iwae_z, mc_z):
    #     for k in range(len(data_expanded)):
    #         print("Sample", k, )
    #         current_data = data_expanded[k].unsqueeze(0)
    #         current_mask = mask_expanded[k].unsqueeze(0)
    #         # current_pathwise_sample = pathwise_sample[k].unsqueeze(0)
    #         current_pathwise_sample = self.onepass.model.encoder.reparam_trick.sample_pathwise((1, iwae_z, mc_z, 1, 10))
    #         for optim in self.optim_list:
    #             optim.zero_grad()
    #         (loss, output_dict) = self.onepass.model(current_data, current_mask, current_pathwise_sample, iwae_z, mc_z, )
    #         loss = loss.mean().backward()
    #         list_grad = list(self.onepass.model.parameters())
    #         list_grad = torch.cat([p.grad.flatten().detach().clone() for p in list_grad])
    #         list_grad_to_compare = torch.cat([grad_weight_to_compare[i][k].flatten() for i in range(len(grad_weight_to_compare))])
    #         print((list_grad - list_grad_to_compare).abs().sum())

    # def single_gradient_test_2(self,grad_weight_to_compare, data_expanded, mask_expanded, pathwise_sample, iwae_z, mc_z):
    #     for k in range(len(data_expanded)):
    #         print("Sample", k, )
    #         current_data = data_expanded[k,0,0].unsqueeze(0)
    #         current_mask = mask_expanded[k,0,0].unsqueeze(0)
    #         current_pathwise_sample = pathwise_sample[k].unsqueeze(0)
    #         for optim in self.optim_list:
    #             optim.zero_grad()
    #         (loss, output_dict) = self.onepass.model.forward_original(current_data, current_mask, None, iwae_z, mc_z, )
    #         loss = loss.mean().backward()
    #         list_grad = list(self.onepass.model.parameters())
    #         list_grad = torch.cat([p.grad.flatten().detach().clone() for p in list_grad])
    #         list_grad_to_compare = torch.cat([grad_weight_to_compare[i][k].flatten() for i in range(len(grad_weight_to_compare))])
    #         print((list_grad - list_grad_to_compare).abs().sum())

    # def average_gradient_test(self,grad_weight_to_compare, data, mask, iwae_z, mc_z):
    #     list_grad_to_compare = torch.cat([grad_weight_to_compare[i].mean(dim = 0).flatten() for i in range(len(grad_weight_to_compare))])
    #     batch_size = data.shape[0]

    #     for optim in self.optim_list:
    #         optim.zero_grad()
    #     (loss, output_dict) = self.onepass.model(data, mask, iwae_z, mc_z, )
    #     loss = loss.mean().backward()
    #     list_grad = list(self.onepass.model.parameters())
    #     list_grad = torch.cat([p.grad.flatten().detach().clone() for p in list_grad])
    #     print("")
    #     print(list_grad.mean())
    #     print(list_grad_to_compare.mean())
    #     print("AVERAGE GRADIENT ERROR", (list_grad - list_grad_to_compare).abs().mean())
    #     print("=================================")

    # def average_gradient_test2(self, grad_weight_to_compare, data, mask, iwae_z, mc_z):
    #     list_grad_to_compare = torch.cat([grad_weight_to_compare[i].mean(dim=0).flatten() for i in range(len(grad_weight_to_compare))])
    #     for optim in self.optim_list:
    #         optim.zero_grad()

    #     (loss, output_dict) = self.onepass.model.forward_original(data, mask, pathwise_sample=None, iwae_sample_z = iwae_z, mc_sample_z =  mc_z, )
    #     loss = loss.mean().backward()
    #     list_grad = list(self.onepass.model.parameters())
    #     list_grad = torch.cat([p.grad.flatten().detach().clone() for p in list_grad])
    #     print(list_grad.mean())
    #     print(list_grad_to_compare.mean())
    #     print("AVERAGE GRADIENT ERROR", (list_grad - list_grad_to_compare).abs().mean())
    #     print("=================================")
        
    def _compute_loss_stateless_model(self, params, buffers, data, mask, weights, iwae_z, mc_z, sample_pathwise ):
        data = data.unsqueeze(0)
        mask = mask.unsqueeze(0)
        sample_pathwise = sample_pathwise.unsqueeze(0)
        output, output_dict = self.fmodel(params, buffers, data, mask, iwae_z, mc_z, sample_pathwise)
        loss_per_sample = -output.sum()*weights
        output_dict["iwae_bound"] = output_dict["iwae_bound"] * weights
        output_dict["vae_bound"] = output_dict["vae_bound"] * weights
        return loss_per_sample.sum(), (loss_per_sample,  output_dict)
        

    def backward_handler(self, sample, loader_train):
        self.fmodel, self.params, self.buffers = make_functional_with_buffers(self.onepass.model)

        in_dims = (None, None, 0, 0, 0, None, None, 0)
        device = next(self.onepass.model.parameters()).device
        data, mask, weights = sample["data"].to(device), sample["mask"].to(device), sample["weights"].to(device)
        batch_size = data.shape[0]
        

        self.ft_compute_grad = grad(self._compute_loss_stateless_model, argnums=0, has_aux=True,)
        self.ft_compute_sample_grad = vmap(self.ft_compute_grad, in_dims)

        iwae_z = self.onepass.iwae_z
        mc_z = self.onepass.mc_z
        latent_dim = self.onepass.model.encoder.latent_dim
        pathwise_sample = self.onepass.model.encoder.reparam_trick.sample_pathwise((batch_size, iwae_z, mc_z, 1, latent_dim)).to(data.device)

        grad_weight_per_example, (loss, output_dict) = self.ft_compute_sample_grad(self.params,
                                                            self.buffers,
                                                            data,
                                                            mask,
                                                            weights,
                                                            iwae_z,
                                                            mc_z,
                                                            pathwise_sample,)


        loss_total = loss.mean()
        output_dict["batch_size"] = batch_size
        batch_sampler = loader_train.batch_sampler
        nb_strata = batch_sampler.nb_strata

        count_grad = torch.stack([torch.count_nonzero(sample["strata"]==i) for i in range(nb_strata)])
        sum_grad_2_per_sample = torch.zeros(batch_size,)
        for i, param in enumerate(self.onepass.model.parameters()):
            param.grad = grad_weight_per_example[i].mean(dim=0).detach()
            if torch.any(torch.isnan(param.grad)):
                logging.warning("Nan in grad")
                assert 1==0
            sum_grad_2_per_sample += (grad_weight_per_example[i].flatten(1)**2).sum(dim =1).detach().cpu()

        sum_grad_2_per_sample = sum_grad_2_per_sample/(sample["weights"].detach().cpu()**2)

        sum_grad = torch.zeros(nb_strata,)
        for strata in range(nb_strata):
            sample_indexes = torch.where(sample["strata"]==strata)[0]
            sum_grad[strata] = sum_grad_2_per_sample[sample_indexes].sum()
            output_dict["sum_grad_2_per_sample_{}".format(strata)] = sum_grad_2_per_sample[sample_indexes].sum().item()
            output_dict["count_grad_{}".format(strata)] = count_grad[strata]
        
        if hasattr(batch_sampler, "fischer_information_approximation") :
            batch_sampler.fischer_information_approximation.update(self.onepass.model)
        batch_sampler.update_grad(sum_grad, count_grad,)   
        

      

        

        return loss_total, output_dict
