from tqdm import tqdm
import torch
try :
    import wandb
except:
    pass


def one_pass(VAE, batch, iwae_z = 1, mc_z = 1, return_dict = False):
    device = next(VAE.parameters()).device
    input = batch['data'].to(device)
    if "mask" in batch.keys():
        mask = batch['mask'].to(device)
    else :
        mask = None
    _output_dict_ = VAE(input, mask = mask, iwae_sample_z = iwae_z, mc_sample_z = mc_z,)
    
    loss = -_output_dict_['iwae_bound']

    if return_dict:
        return loss, _output_dict_
    else :
        return loss


def train_epoch(VAE, train_loader, args, epoch = -1,):
    tmp_kl = 0
    tmp_likelihood = 0
    tmp_vae_elbo = 0
    tmp_iwae_elbo = 0
    obs_in_epoch = 0

    pbar = tqdm(enumerate(train_loader))
    for i, batch in tqdm(enumerate(train_loader)):
        

        VAE.train()
        batch_size = batch['data'].shape[0]
        VAE.optim_decoder.zero_grad()
        VAE.optim_encoder.zero_grad()
        loss, output_dict = one_pass(VAE = VAE, batch = batch, iwae_z= args["iwae_z"], mc_z=args["mc_z"], return_dict=True)
        loss.backward()
        VAE.optim_decoder.step()
        VAE.optim_encoder.step()


        tmp_kl += torch.sum(torch.mean(output_dict['kl'],dim=0)).item()
        tmp_likelihood += torch.sum(torch.mean(output_dict['likelihood'], dim=0)).item()
        tmp_vae_elbo += output_dict['vae_bound'].item() * batch_size
        tmp_iwae_elbo += output_dict['iwae_bound'].item() * batch_size
        obs_in_epoch += batch_size

        if i % 100 == 0:
            current_kl = tmp_kl / obs_in_epoch
            current_likelihood = tmp_likelihood / obs_in_epoch
            current_vae_elbo = tmp_vae_elbo / obs_in_epoch
            current_iwae_elbo = tmp_iwae_elbo / obs_in_epoch
            desc = "KL: {:.2f} | log p(x): {:.2f} | VAE ELBO: {:.2f} | IWAE ELBO: {:.2f}".format(current_kl, current_likelihood, current_vae_elbo, current_iwae_elbo)
            # pbar.set_description(desc, refresh=True)
            pbar.write(desc, )


    print(
        "epoch {0}/{1}, train VAE ELBO: {2:.2f}, train IWAE bound: {3:.2f}, train likelihod: {4:-2f}, train KL: {5:.2f}"
            .format(epoch, args["nb_epoch"], tmp_vae_elbo / obs_in_epoch, tmp_iwae_elbo / obs_in_epoch,
                    tmp_likelihood / obs_in_epoch, tmp_kl / obs_in_epoch))

    if args["use_wandb"]:
        wandb.log({
            "epoch": epoch, "train VAE ELBO": tmp_vae_elbo / obs_in_epoch,
            'train IWAE bound': tmp_iwae_elbo / obs_in_epoch,
            "train likelihod": tmp_likelihood / obs_in_epoch, "train KL": tmp_kl / obs_in_epoch})
