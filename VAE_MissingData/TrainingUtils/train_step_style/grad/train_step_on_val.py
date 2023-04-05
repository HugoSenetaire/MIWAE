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
    def __init__(self, onepass, optim_list, scheduler_list = [],  delta = 0.5, **kwargs):
        super().__init__(onepass, optim_list = optim_list, scheduler_list = scheduler_list,)
        self.count = 0
        assert delta >=0 and delta <=1, "delta should be between 0 and 1"
        self.delta = delta
        self.proportion = None



    
    def calculate_new_proportion(self, loader_train, loader_val):
        batch_sampler_train = loader_train.batch_sampler
        batch_sampler_val = loader_val.batch_sampler
        self.count+=1
        if self.count>0 and self.count% 50 ==0:
            # print("proportion before", loader_train.batch_sampler.p_i)      

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
                    loss = self.onepass(sample)
                    loss.mean().backward()
                    for param in self.onepass.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_2[current_strata] += torch.sum(param.grad.flatten()**2).detach().clone()
                    nb_grad[current_strata] += 1 


            grad_2 = grad_2 / nb_grad
            new_proportion = batch_sampler_train.w_i * grad_2.sqrt().detach().cpu()
            new_proportion = new_proportion / new_proportion.sum()
            if self.proportion is not None:
                self.proportion = self.proportion * (1-self.delta) + new_proportion * self.delta
            else :
                self.proportion = new_proportion
            batch_sampler_train.p_i = self.proportion


    def __call__(self, sample, loader_train, loader_val = None, take_step = True, proportion_calculation = False ):

        output = super().__call__(sample, loader_train, loader_val = loader_val, take_step = take_step)
        if proportion_calculation :
            self.calculate_new_proportion(loader_train, loader_val) 
        return output

                    
