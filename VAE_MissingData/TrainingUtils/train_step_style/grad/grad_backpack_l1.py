
from ..trainer_step_default import TrainerStepDefault
from backpack.extensions import BatchGrad
from backpack import backpack, extend

import logging
import torch


def get_gradnorm_from_backpack_batch(model, strata_rep, sampler, weights):
    list_param = list(sampler.parameter_iterator.__iter__(model)) #TODO : If this is just done once, could be faster
    nb_strata = sampler.nb_strata
    sum_grad = torch.zeros(nb_strata, dtype = torch.float32,)
    count_grad = torch.stack([torch.sum(strata_rep==i) for i in range(nb_strata)])
    repetition = sampler.repetition

    for i in range(nb_strata):
        sample_indexes = torch.where(strata_rep==i) # Sample indexes corresponding to strata_rep[i]
        for name, p in list_param:
            if p.grad is not None :
                aux_grad = None
                for k in range(0,repetition):
                    if aux_grad is None :
                        aux_grad = p.grad_batch.detach()[sample_indexes].cpu()
                    else :
                        aux_grad += p.grad_batch.detach()[sample_indexes].cpu()
                    if hasattr(sampler, 'fischer_information_approximation'):
                        sum_grad[i] += torch.sum(aux_grad.pow(2), dim =0).flatten().dot(sampler.fischer_information_approximation.reverse_fischer_information[name].flatten())
                    else :
                        sum_grad[i] += torch.sum(aux_grad.pow(2),)
        current_strata_weight = weights[sample_indexes[0]][0]        
        sum_grad[i] /= current_strata_weight**2
    return sum_grad, count_grad


class TrainerStepBackpackBatch(TrainerStepDefault):
    def __init__(self, onepass,  optim_list, scheduler_list = []) -> None:
        super().__init__(onepass,
                        optim_list,
                        scheduler_list = scheduler_list,
                        )

        self.onepass.model = extend(self.onepass.model)


    def backward_handler(self, sample, loader_train):
        strata_per_sample = sample["strata"]
        sig_f = torch.zeros(loader_train.batch_sampler.nb_strata, dtype = torch.float32)
       
        loss = self.onepass(sample,)
        
        loss = torch.dot(loss, sample["weights"])
        
        with backpack(BatchGrad()):
            loss.backward()

        batch_sampler = loader_train.batch_sampler
        if hasattr(batch_sampler, "fischer_information_approximation") :
            batch_sampler.fischer_information_approximation.update(self.onepass.model)
        sig_f, count_grad = get_gradnorm_from_backpack_batch(self.onepass.model, strata_per_sample, batch_sampler, sample["weights"])
        
        batch_sampler.update_grad(sig_f, count_grad)   
        return loss
