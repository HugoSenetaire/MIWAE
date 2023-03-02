from tqdm import tqdm
import torch
from backpack.extensions import BatchGrad, BatchL2Grad, Variance
from backpack import backpack
from .evaluation import eval
from .train_step_style import trainer_step_default
from ..loss_handler import WeightsMultiplication
import copy
import numpy as np
try :
    import wandb
except:
    pass





def train_epoch(trainer_step,
                train_loader,
                args,
                val_loader = None,
                epoch = -1,
                writer = None,
                test_loader = None,
                eval_iter = 100,
                save_image_iter = 300,
                best_valid_log_likelihood = -float('inf')):
    tmp_kl = []
    tmp_likelihood = []
    tmp_vae_elbo = []
    tmp_iwae_elbo = []
    obs_in_epoch = []

    pbar = tqdm(enumerate(train_loader))
    for i, batch in pbar:
        iteration = epoch * len(train_loader) + i
        _, output_dict = trainer_step(batch, loader_train = train_loader)
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'update_p_i'):
            train_loader.batch_sampler.update_p_i()  

        writer.add_scalar('train/norm_grad', output_dict['norm_grad'], iteration)
        if iteration%10 == 0 :
            grads = []
            train_loader_aux = copy.deepcopy(train_loader)
            print("Calculating variance of gradient on train_loader")
            for k, sample in enumerate(iter(train_loader_aux)):
                loss_per_instance, output_dict = trainer_step(sample = sample, loader_train = train_loader_aux, take_step = False, proportion_calculation = True)
                grads += [torch.cat([p.grad.flatten() for p in trainer_step.onepass.model.parameters() if p.grad is not None])]
                if k==10:
                    break
            variance_grad = torch.var(torch.stack(grads), dim = 0).sum(-1)
            output_dict['variance_grad_train'] = variance_grad
            writer.add_scalar('train/variance_grad', variance_grad, iteration)


            if val_loader is not None :
                grads = []
                print("Calculating variance of gradient on val")
                if hasattr(val_loader, 'batch_sampler') and hasattr(val_loader.batch_sampler, 'update_p_i'):
                    val_loader.batch_sampler.p_i = train_loader.batch_sampler.p_i
                for k, sample in enumerate(iter(val_loader)):
                    loss_per_instance, output_dict = trainer_step(sample = sample, loader_train = val_loader, take_step = False)
                    grads += [torch.cat([p.grad.flatten() for p in trainer_step.onepass.model.parameters() if p.grad is not None])]
                    if k==10:
                        break
                variance_grad = torch.var(torch.stack(grads), dim = 0).sum(-1)
                output_dict['variance_grad_val'] = variance_grad
                writer.add_scalar('val/variance_grad', variance_grad, iteration)
            torch.cuda.empty_cache() 

        tmp_kl.append(output_dict['kl'].sum().item())
        tmp_likelihood.append(output_dict['likelihood'].sum().item())
        tmp_vae_elbo.append(output_dict['vae_bound'].sum().item())
        tmp_iwae_elbo.append(output_dict['iwae_bound'].sum().item())
        obs_in_epoch.append(output_dict['batch_size'])

        writer.add_scalar('train/KL',torch.tensor(np.sum(tmp_kl[-10:])/np.sum(obs_in_epoch[-10:])) , iteration)
        print("KL", np.sum(tmp_kl[-10:])/np.sum(obs_in_epoch[-10:]))
        writer.add_scalar('train/likelihood',torch.tensor(np.sum(tmp_likelihood[-10:])/np.sum(obs_in_epoch[-10:])), iteration)
        print("likelihood", np.sum(tmp_likelihood[-10:])/np.sum(obs_in_epoch[-10:]))
        writer.add_scalar('train/vae_elbo',torch.tensor(np.sum(tmp_vae_elbo[-10:])/np.sum(obs_in_epoch[-10:])), iteration)
        print("vae_elbo", np.sum(tmp_vae_elbo[-10:])/np.sum(obs_in_epoch[-10:]))
        writer.add_scalar('train/iwae_elbo',torch.tensor(np.sum(tmp_iwae_elbo[-10:])/np.sum(obs_in_epoch[-10:])), iteration)
        print("iwae_elbo", np.sum(tmp_iwae_elbo[-10:])/np.sum(obs_in_epoch[-10:]))
        if iteration % 100 == 0:
            current_kl = np.sum(tmp_kl) / np.sum(obs_in_epoch)
            current_likelihood = np.sum(tmp_likelihood) / np.sum(obs_in_epoch)
            current_vae_elbo = np.sum(tmp_vae_elbo) / np.sum(obs_in_epoch)
            current_iwae_elbo = np.sum(tmp_iwae_elbo) / np.sum(obs_in_epoch)
            desc = "KL: {:.2f} | log p(x): {:.2f} | VAE ELBO: {:.2f} | IWAE ELBO: {:.2f}".format(current_kl, current_likelihood, current_vae_elbo, current_iwae_elbo)
            pbar.write(desc, )
        
        if (iteration+1)%eval_iter == 0 and writer is not None:
            print("Evaluating")
            eval(iteration= iteration, one_pass=trainer_step.onepass, val_loader=test_loader,
                args=args, best_valid_log_likelihood=best_valid_log_likelihood, writer=writer, sample=True)
            # args=args, best_valid_log_likelihood=best_valid_log_likelihood, writer=writer, sample= ((iteration+1)%save_image_iter == 0))


        for key in output_dict.keys():
            # if key not in ['batch_size', 'kl', 'likelihood', 'vae_bound', 'iwae_bound']:
                # output_dict[key] = output_dict[key].mean(dim=0)
            try :
                writer.add_scalars('train/{}'.format(key), output_dict[key], iteration)
            except AttributeError as e :
                pass

        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'update_p_i'):
            for k in range(len(train_loader.batch_sampler.p_i)):
                writer.add_scalar('train/p_i_{}'.format(k), train_loader.batch_sampler.p_i[k], iteration)
                writer.add_scalar('train/w_i_{}'.format(k), train_loader.batch_sampler.w_i[k], iteration)
                writer.add_scalar('train/n_i_{}'.format(k), train_loader.batch_sampler.n_i[k], iteration)
        del batch

        for key in output_dict.keys():
            if key.startswith("count_grad") or key.startswith("sum_grad"):
                writer.add_scalar('train/{}'.format(key), output_dict[key], iteration)

    print(
        "epoch {0}/{1}, train VAE ELBO: {2:.2f}, train IWAE bound: {3:.2f}, train likelihod: {4:-2f}, train KL: {5:.2f}"
            .format(epoch, args["nb_epoch"], np.sum(tmp_vae_elbo) / np.sum(obs_in_epoch),
                        np.sum(tmp_iwae_elbo) / np.sum(obs_in_epoch),
                        np.sum(tmp_likelihood) / np.sum(obs_in_epoch), 
                        np.sum(tmp_kl) / np.sum(obs_in_epoch)))

   