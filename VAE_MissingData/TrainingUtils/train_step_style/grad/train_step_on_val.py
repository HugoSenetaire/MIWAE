import torch
import numpy as np
from ..trainer_step_default import TrainerStepDefault
import tqdm
def add_to_dic(dic, key, value):
    if key not in dic.keys():
        dic[key] = [value]
    else :
        dic[key].append(value)
    return dic

class TrainerStepProportionOnVal(TrainerStepDefault):
    """
    Class that handles how one should create 
    """
    def __init__(self, onepass, optim_list, scheduler_list = [],):
        super().__init__(onepass, optim_list = optim_list, scheduler_list = scheduler_list,)
        self.count = 0


    
    def calculate_new_proportion(self, loader_train, loader_val):
        if hasattr(loader_train, "batch_sampler") and hasattr(loader_val, "batch_sampler") :
            
            batch_sampler_train = loader_train.batch_sampler
            batch_sampler_val = loader_val.batch_sampler
            self.count+=1
            if self.count>0 and self.count% 20 ==0:
                nb_strata = len(batch_sampler_train.p_i)
                grad_2 = torch.zeros(nb_strata, device = self.device)
                nb_grad = torch.zeros(nb_strata, device = self.device)
                for current_strata in tqdm.tqdm(range(nb_strata)):
                    indexes = np.where(batch_sampler_val.index_strata_correspondence == current_strata)[0]
                    current_indexes = np.random.choice(indexes, size = min(1000, len(indexes)), replace = False)
                    for i in current_indexes:
                        sample = loader_val.dataset.__getitem__(i)
                        strata = sample["strata"]
                        assert strata == current_strata
                        for key in sample.keys():
                            sample[key] = torch.tensor(sample[key],).unsqueeze(0).to(self.device)
                        self.onepass.model.zero_grad()
                        loss, _ = self.onepass(sample)
                        loss.mean().backward()
                        for param in self.onepass.model.parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_2[current_strata] += torch.sum(param.grad.flatten()**2).detach().clone()
                        nb_grad[current_strata] += 1 


                grad_2 = grad_2 / nb_grad
                # batch_sampler_train.sig_f_running = grad_2
                # batch_sampler_train.count_grad = nb_grad
                # loader_train.batch_sampler.update_p_i()
                new_proportion = batch_sampler_train.w_i * grad_2.sqrt()
                new_proportion = new_proportion / new_proportion.sum()
                batch_sampler_train.p_i = new_proportion


    def __call__(self, sample, loader_train, loader_val = None, take_step = True, proportion_calculation = False ):
        output = super().__call__(sample, loader_train, loader_val = loader_val, take_step = take_step)
        if proportion_calculation :
            self.calculate_new_proportion(loader_train, loader_val)        
        return output

                    
