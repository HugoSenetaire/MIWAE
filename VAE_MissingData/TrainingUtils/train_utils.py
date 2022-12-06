import torch


class Onepass():
    def __init__(self, model, iwae_z = 1, mc_z = 1,):
        self.model = model
        self.device = next(model.parameters()).device
        self.loss_function = torch.nn.Identity()
        self.iwae_z = iwae_z
        self.mc_z = mc_z

    def __call__(self, sample, return_dict = False):
        input = sample['data'].to(self.device)
        if "mask" in sample.keys():
            mask = sample['mask'].to(self.device)
        else :
            mask = None
        if return_dict :
            out_bound, _output_dict_ = self.model(input, mask = mask, iwae_sample_z = self.iwae_z, mc_sample_z = self.mc_z, return_dict = return_dict)
            _output_dict_['batch_size'] = input.shape[0]
        else :
            out_bound = self.model(input, mask = mask, iwae_sample_z = self.iwae_z, mc_sample_z = self.mc_z, return_dict = return_dict)
        loss_per_instance = -out_bound
        if return_dict:
            return loss_per_instance, _output_dict_
        else :
            return loss_per_instance
