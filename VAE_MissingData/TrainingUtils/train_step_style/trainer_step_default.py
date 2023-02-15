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
        return loss_per_instance.sum(), output_dict

    # def average_gradient_test(self, grad_weight_to_compare, data_expanded, mask_expanded, pathwise_sample, iwae_z, mc_z):
    #     list_grad_to_compare = grad_weight_to_compare
    #     batch_size = data_expanded.shape[0]
    #     data_expanded = data_expanded.unsqueeze(1).unsqueeze(1).expand(-1, iwae_z, mc_z, -1, -1, -1)
    #     mask_expanded = mask_expanded.unsqueeze(1).unsqueeze(1).expand(-1, iwae_z, mc_z, -1, -1, -1)
    #     pathwise_sample = self.onepass.model.encoder.reparam_trick.sample_pathwise((batch_size, iwae_z, mc_z, 1, 10))

    #     for optim in self.optim_list:
    #         optim.zero_grad()
    #     (loss, output_dict) = self.onepass.model(data_expanded, mask_expanded, pathwise_sample, iwae_z, mc_z, )
    #     loss = loss.mean().backward()
    #     list_grad = list(self.onepass.model.parameters())
    #     list_grad = torch.cat([p.grad.flatten().detach().clone() for p in list_grad])
    #     print(list_grad.mean())
    #     print(list_grad_to_compare.mean())
    #     print("AVERAGE GRADIENT ERROR", (list_grad - list_grad_to_compare).abs().sum())
    #     print("=================================")

    # def average_gradient_test2(self, grad_weight_to_compare, data, mask, pathwise_sample, iwae_z, mc_z):
    #     list_grad_to_compare = grad_weight_to_compare
    #     for optim in self.optim_list:
    #         optim.zero_grad()

    #     (loss, output_dict) = self.onepass.model.forward_original(data, mask, None, iwae_z, mc_z, )
    #     loss = loss.mean().backward()
    #     list_grad = list(self.onepass.model.parameters())
    #     list_grad = torch.cat([p.grad.flatten().detach().clone() for p in list_grad])
    #     print(list_grad.mean())
    #     print(list_grad_to_compare.mean())
    #     print("AVERAGE GRADIENT ERROR", (list_grad - list_grad_to_compare).abs().sum())
    #     print("=================================")

    def __call__(self, sample, loader_train, ):
        self.onepass.model.train()
        for optim in self.optim_list:
            optim.zero_grad()



        loss, output_dict = self.backward_handler(sample, loader_train=loader_train)
        # list_grad = list(self.onepass.model.parameters())
        # list_grad = torch.cat([p.grad.flatten().detach().clone() for p in list_grad])

        # iwae_z = self.onepass.iwae_z
        # mc_z = self.onepass.mc_z
        # self.average_gradient_test(list_grad, sample["data"], sample["mask"], pathwise_sample = None, iwae_z=iwae_z, mc_z=mc_z)
        # self.average_gradient_test2(list_grad, sample["data"], sample["mask"], pathwise_sample=None, iwae_z=iwae_z, mc_z=mc_z)

        # for optim in self.optim_list:
        #     optim.zero_grad()

        # loss_per_instance_2, output_dict_2 = self.onepass(sample = sample, return_dict=True)
        # loss_per_instance_2 = loss_per_instance_2.mean().backward()
        # list_grad2 = list(self.onepass.model.parameters())
        # list_grad2 = torch.cat([p.grad.flatten().detach().clone() for p in list_grad2])

        

        # print("DIFF IN GRADIENTS", torch.norm(list_grad - list_grad2))
        # for key in output_dict.keys():
        #     if key in output_dict_2.keys() and isinstance(output_dict[key], torch.Tensor):
        #         # print(output_dict[key].shape, output_dict_2[key].shape)
        #         try :
        #             print(key, torch.norm(output_dict[key].mean(dim=0) - output_dict_2[key].mean(dim=0)))
        #         except TypeError:
        #             print(key, torch.norm(output_dict[key] - output_dict_2[key]))
            
        for optim in self.optim_list:
            optim.step()
        for scheduler in self.scheduler_list:
            scheduler.step()

        

        return loss, output_dict

