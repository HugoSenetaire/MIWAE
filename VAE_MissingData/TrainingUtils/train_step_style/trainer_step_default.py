import torch
from torch.autograd import grad

class TrainerStepDefault():
    """
    Class that handles how one should create 
    """
    def __init__(self, onepass, optim_list, scheduler_list = [],) -> None:
        self.onepass = onepass,
        if isinstance(self.onepass, tuple):
            self.onepass = self.onepass[0]
        self.optim_list = optim_list
        self.scheduler_list = scheduler_list
        self.device = next(self.onepass.model.parameters()).device

    def backward_handler(self, sample, loader_train):
        loss_per_instance, output_dict = self.onepass(sample = sample, return_dict=True)
        loss_per_instance.mean().backward()
        return loss_per_instance.mean(), output_dict


    def __call__(self, sample, loader_train, loader_val = None, take_step = True, proportion_calculation = False ):
        self.onepass.model.train()
        for optim in self.optim_list:
            optim.zero_grad()
        
        loss, output_dict = self.backward_handler(sample, loader_train=loader_train)
       
        norm_grad = torch.norm(torch.stack([torch.norm(p.grad.flatten()) for p in self.onepass.model.parameters() if p.grad is not None]),)
        output_dict["norm_grad"] = norm_grad
            
        if take_step :
            for optim in self.optim_list:
                optim.step()
            for scheduler in self.scheduler_list:
                scheduler.step()

        

        return loss, output_dict

