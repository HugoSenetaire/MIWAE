import torch
import torchvision
try :
    import wandb
except:
    pass
from .train import one_pass
import os

def save_samples(VAE, args, epoch = -1):
    with torch.no_grad():
        VAE.eval()
        device = next(VAE.parameters()).device

        samples, _ = VAE.sample_from_prior(latents = VAE.latents_for_eval.to(device),)

        if args["use_wandb"]:
            grid = torchvision.utils.make_grid(samples, nrows = 10)
            wandb.log({"samples_prior_{}".format(epoch): wandb.Image(grid)})
        else :
            torchvision.utils.save_image(samples.cpu().detach(), os.path.join(args["samples_dir"], "sample_prior_{}.png".format(epoch)),nrow = 10)

def save_samples_cond(VAE, args, val_loader, epoch = -1, n_samples = 7, nb_inputs = 10):
        device = next(VAE.parameters()).device

        inputs = next(iter(val_loader))['data'][:nb_inputs].to(device)
        masks = next(iter(val_loader))['mask'][:nb_inputs].to(device)
        masked_input = inputs * masks + (1 - masks)*0.1
        samples = VAE.sample_from_input(inputs, masks, n_samples = n_samples)

        inputs_expanded = inputs.unsqueeze(0).expand(n_samples, -1, -1, -1, -1).flatten(0, 1)
        masks_expanded = masks.unsqueeze(0).expand(n_samples, -1, -1, -1, -1).flatten(0, 1)
        samples_mix = samples * (1-masks_expanded) + masks_expanded*inputs_expanded
        # samples = samples.reshape(n_samples, nb_inputs, *samples.shape[1:]).permute(1,0,2,3,4).flatten(0,1)
        to_save = torch.cat([inputs, masks, masked_input, samples], dim=0)
        to_save_mix = torch.cat([inputs, masks, masked_input, samples_mix], dim=0)
        if args["use_wandb"]:
            grid = torchvision.utils.make_grid(to_save, nrows = nb_inputs)
            wandb.log({"samples_cond_{}".format(epoch): wandb.Image(grid)})
            grid2 = torchvision.utils.make_grid(to_save_mix, nrows = nb_inputs)
            wandb.log({"samples_cond_mix_{}".format(epoch): wandb.Image(grid2)})
        else :
            torchvision.utils.save_image(to_save.cpu().detach(), os.path.join(args["samples_dir"], "sample_cond_{}.png".format(epoch)),nrow = nb_inputs)
            torchvision.utils.save_image(to_save_mix.cpu().detach(), os.path.join(args["samples_dir"], "sample_cond_mix_{}.png".format(epoch)),nrow = nb_inputs)

def eval(epoch, VAE, val_loader, args, best_valid_log_likelihood = -float('inf'),):
    with torch.no_grad():
        VAE.eval()

        valid_log_like = 0
        valid_iwae_log_like = 0
        valid_obs = 0
        for j, batch_input in enumerate(val_loader):
            batch_size = batch_input['data'].shape[0]
            valid_obs += batch_size

            # for now I am not using IWAE bound
            _, output_dict = one_pass(VAE, batch_input, iwae_z = args["iwae_z_test"], mc_z = args["mc_z_test"], return_dict = True)
            valid_iwae = output_dict['iwae_bound'] * batch_size
            valid_elbo = output_dict['vae_bound'] * batch_size
            valid_log_like += valid_elbo
            valid_iwae_log_like += valid_iwae

        # compute the final avg elbo for the validation set
        avg_valid = valid_log_like / valid_obs
        print('Validation log p(x): ', avg_valid)
        if args["use_wandb"]:
            wandb.log({'Validation log p(x)': avg_valid})
        save_samples(VAE, args, epoch=epoch)
        save_samples_cond(VAE, args, val_loader, epoch=epoch)
        
        if avg_valid > best_valid_log_likelihood:
            best_valid_log_likelihood = avg_valid
            
            # save VAE
            torch.save(VAE.state_dict(),
                       os.path.join(os.path.join(args["model_dir"], args["name_experiment"]),"model" +'.pt'))

