from tqdm import tqdm
import torch
from backpack.extensions import BatchGrad, BatchL2Grad, Variance
from backpack import backpack
from .evaluation import eval
from ..loss_handler import WeightsMultiplication
try :
    import wandb
except:
    pass


def train_step(one_pass, sample, batch_sampler, return_dict = True, args_dict = None):
    VAE = one_pass.model
    VAE.train()
    VAE.optim_decoder.zero_grad()
    VAE.optim_encoder.zero_grad()
    # loss, output_dict = one_pass(VAE = VAE, batch = batch, iwae_z= args["iwae_z"], mc_z=args["mc_z"], return_dict=True)
    loss_per_instance, output_dict = one_pass(sample = sample, return_dict=return_dict)
    if hasattr(batch_sampler, 'weights_list'):
        loss = loss_per_instance.dot(batch_sampler.weights_list)
    else :
        loss = loss_per_instance.sum()

    if args_dict is not None and args_dict["use_backpack"] :
        with backpack(BatchGrad(), BatchL2Grad(), Variance()):
            loss.backward()
    else :
        loss.backward()

    VAE.optim_decoder.step()
    VAE.optim_encoder.step()

    if return_dict:
        return loss, output_dict
    else :
        return loss



def train_epoch(one_pass,
                train_loader,
                args,
                epoch = -1,
                writer = None,
                test_loader = None,
                eval_iter = 100,
                save_image_iter = 300,
                best_valid_log_likelihood = -float('inf')):
    tmp_kl = 0
    tmp_likelihood = 0
    tmp_vae_elbo = 0
    tmp_iwae_elbo = 0
    obs_in_epoch = 0

    pbar = tqdm(enumerate(train_loader))
    for i, batch in pbar:
        iteration = epoch * len(train_loader) + i
        if hasattr(train_loader, 'batch_sampler') :
            batch_sampler = train_loader.batch_sampler
        else :
            batch_sampler = None
        _, output_dict = train_step(one_pass, batch, return_dict = True, batch_sampler = batch_sampler)
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'update_p_i'):
            train_loader.batch_sampler.update_p_i()  


        tmp_kl += torch.sum(torch.mean(output_dict['kl'],dim=0)).item()
        tmp_likelihood += torch.sum(torch.mean(output_dict['likelihood'], dim=0)).item()
        tmp_vae_elbo += output_dict['vae_bound'].item() * output_dict['batch_size']
        tmp_iwae_elbo += output_dict['iwae_bound'].item() * output_dict['batch_size']
        obs_in_epoch += output_dict['batch_size']

        if iteration % 100 == 0:
            current_kl = tmp_kl / obs_in_epoch
            current_likelihood = tmp_likelihood / obs_in_epoch
            current_vae_elbo = tmp_vae_elbo / obs_in_epoch
            current_iwae_elbo = tmp_iwae_elbo / obs_in_epoch
            desc = "KL: {:.2f} | log p(x): {:.2f} | VAE ELBO: {:.2f} | IWAE ELBO: {:.2f}".format(current_kl, current_likelihood, current_vae_elbo, current_iwae_elbo)
            pbar.write(desc, )
        
        if (iteration+1)%eval_iter == 0 and writer is not None:
            eval(iteration= iteration, one_pass=one_pass, val_loader=test_loader,
            args=args, best_valid_log_likelihood=best_valid_log_likelihood, writer=writer, sample= ((iteration+1)%save_image_iter == 0))


        for key in output_dict.keys():
            # if key not in ['batch_size', 'kl', 'likelihood', 'vae_bound', 'iwae_bound']:
                # output_dict[key] = output_dict[key].mean(dim=0)
            try :
                writer.add_scalars('train/{}'.format(key), output_dict[key], iteration)
            except AttributeError as e :
                pass


    print(
        "epoch {0}/{1}, train VAE ELBO: {2:.2f}, train IWAE bound: {3:.2f}, train likelihod: {4:-2f}, train KL: {5:.2f}"
            .format(epoch, args["nb_epoch"], tmp_vae_elbo / obs_in_epoch, tmp_iwae_elbo / obs_in_epoch,
                    tmp_likelihood / obs_in_epoch, tmp_kl / obs_in_epoch))

    if args["use_wandb"]:
        wandb.log({
            "epoch": epoch, "train VAE ELBO": tmp_vae_elbo / obs_in_epoch,
            'train IWAE bound': tmp_iwae_elbo / obs_in_epoch,
            "train likelihod": tmp_likelihood / obs_in_epoch, "train KL": tmp_kl / obs_in_epoch})
