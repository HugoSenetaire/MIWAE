from ..trainer_step_default import TrainerStepDefault
from torch.autograd import grad

import logging
import torch


    
def transpose(sample,):
    keys = sample.keys()
    list_of_sample = []
    for i in range(len(sample["data"])):
        dic = {}
        for key in keys:
            dic[key] = sample[key][i].unsqueeze(0)
        list_of_sample.append(dic)
    
    return list_of_sample


def calculate_single_grad(list_parameter, current_sample, sig_f, previous_grad = {},):
    current_strata = current_sample["strata"]

    for name, p in list_parameter:
        if p.grad is not None :
            if torch.isnan(p.grad).any():
                assert False
            if name not in previous_grad.keys():
                sig_f[current_strata] += torch.sum((p.grad/current_sample["weights"])**2).detach().cpu()   # TODO : Can be accelaerated by using the fact that the weights                                                                        
                                                                                                            # are the same for all the sample in a single strata
            else :
                sig_f[current_strata] += torch.sum(((p.grad-previous_grad[name])/current_sample["weights"])**2).detach().cpu()

            previous_grad[name] = p.grad.detach().clone()

    return previous_grad, sig_f


class TrainerStepManualGradNorm(TrainerStepDefault):
    def __init__(self, onepass,  optim_list, scheduler_list = [], **kwargs) -> None:
        super().__init__(onepass,
                        optim_list,
                        scheduler_list = scheduler_list,
                        )


    def backward_handler(self, sample, loader_train):
        batch_sampler=loader_train.batch_sampler
        nb_strata = len(batch_sampler.strata)
        transpose_sample = transpose(sample)
        sig_f = torch.zeros(nb_strata, device = "cpu")
        count_grad = torch.stack([torch.count_nonzero(sample["strata"]==i) for i in range(nb_strata)])
        list_parameter = list(self.onepass.model.named_parameters())
        loss_total = torch.tensor(0., device = self.device)
        device = next(self.onepass.model.parameters()).device
        previous_grad = {}
        output_dict = {}
        for i, current_sample in enumerate(transpose_sample):
            current_sample["weights"] = current_sample["weights"].to(device)
            loss, current_dict = self.onepass(current_sample, return_dict = True)
            loss = loss*current_sample["weights"]
            loss_total += loss.detach().cpu().item()
            loss.backward(retain_graph=False)
            current_dict["iwae_bound"] = current_dict["iwae_bound"] * current_sample["weights"] 
            current_dict["vae_bound"] = current_dict["vae_bound"] * current_sample["weights"]
            previous_grad, sig_f = calculate_single_grad(list_parameter=list_parameter, current_sample=current_sample, sig_f=sig_f, previous_grad=previous_grad )
            for key in current_dict.keys():
                if key in output_dict.keys():
                    output_dict[key] += [current_dict[key]]
                else :
                    output_dict[key] = [current_dict[key]]
        

        for p in self.onepass.model.parameters():
            if p.grad is not None :
                p.grad /= sample["data"].shape[0]
        for key in output_dict.keys():
            if isinstance(output_dict[key][0], torch.Tensor):
                output_dict[key] = torch.stack(output_dict[key])

        output_dict["batch_size"] = sample["data"].shape[0]
        batch_sampler.update_grad(sig_f, count_grad)  

        for strata in range(nb_strata):
            output_dict["sum_grad_2_per_sample_{}".format(strata)] = sig_f[strata].item()
            output_dict["count_grad_{}".format(strata)] = count_grad[strata]
        
        
        
        if hasattr(loader_train, "fischer_information_approximation") :
            loader_train.fischer_information_approximation.update(self.onepass.model)

        return loss_total, output_dict