import torch
import numpy as np
import random
import os
import time 
from default_args import default_args
from MissingDataDataset import get_dataset
from VAE_MissingData.get_miwae import get_miwae, get_decoder, get_encoder, get_networks, get_decoder_mask
from VAE_MissingData.TrainingUtils import train_epoch, eval, Onepass, get_trainer_step
from VAE_MissingData.StratifiedSGDforMissingData import get_dataloader
from backpack import extend
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


import warnings


def analyze_index_strata_correspondence(train_loader, args_dict, writer,):
     if hasattr(train_loader, "batch_sampler"):
        if hasattr(train_loader.batch_sampler, "index_strata_correspondence"):
            current_index_strata_correspondence = train_loader.batch_sampler.index_strata_correspondence
            nb_strata = np.unique(current_index_strata_correspondence).shape[0]
            mask_dim = np.prod(train_loader.dataset.__getitem__(0)["mask"].shape[1:])

            nb_missing_data = {}
            targets = {}
            nb_targets = np.unique(np.stack([train_loader.dataset.__getitem__(index)["target"] for index in range(len(train_loader.dataset))])).shape[0]
            for i in range(nb_strata):
                indexes = np.where(current_index_strata_correspondence == i)[0]
                current_missing = torch.stack([1-train_loader.dataset.__getitem__(index)["mask"] for index in indexes])
                current_target = torch.stack([train_loader.dataset.__getitem__(index)["target"] for index in indexes])
                nb_missing_data[i] = (1 - current_missing).flatten(1).sum(1)
                targets[i] = current_target

            nb_lines = int(np.ceil(nb_strata/5.))
            nb_cols = 5
            fig, axs = plt.subplots(nrows=nb_lines, ncols=nb_cols, figsize=(nb_cols *4, nb_lines*4))
            bins = np.linspace(0, mask_dim, 101)
            for i in range(nb_strata):
                if nb_lines > 1 :
                    axs[i//5, i%5].hist(nb_missing_data[i].numpy(), bins=bins,)
                    # axs[i//5, i%5].set_xlabel("Number of missing data")
                    # axs[i//5, i%5].set_ylabel("Number of samples")
                    axs[i//5, i%5].set_title("Histogram strata {}".format(i))
                else :
                    axs[i%5].hist(nb_missing_data[i].numpy(), bins=bins,)
            plt.title("Histogram nb_missing")
            plt.savefig(os.path.join(args_dict["log_dir"], "Histogram_nb_missing.png"))
            writer.add_figure("Histogram nb_missing", plt.gcf())
            plt.close()

            fig, axs = plt.subplots(nrows=nb_lines, ncols=nb_cols, figsize=(nb_cols *4, nb_lines*4))
            bins_target = np.arange(0, nb_targets+1, 1)-0.5
            for i in range(nb_strata):
                if nb_lines>1 :
                    axs[i//5, i%5].hist(targets[i].numpy(), bins=bins_target, )
                    # axs[i//5, i%5].set_xticks(np.arange(nb_targets), np.arange(nb_targets))
                    # axs[i//5, i%5].set_xlabel("Target")
                    # axs[i//5, i%5].set_ylabel("Number of samples")
                    axs[i//5, i%5].set_title("Histogram strata {}".format(i))
                else :
                    axs[i%5].hist(targets[i].numpy(), bins=bins_target, )
            plt.title("Histogram target".format(i))
            plt.savefig(os.path.join(args_dict["log_dir"], "Histogram_target.png"))
            writer.add_figure("Histogram target".format(i), plt.gcf())
            plt.close()

            

def create_path(args_dict):
    args_dict["model_dir"] = os.path.join(args_dict["root_dir"], "weights")
    if not os.path.exists(args_dict["model_dir"]):
        os.makedirs(args_dict["model_dir"])

    if "name_experiment" not in args_dict.keys() or args_dict["name_experiment"] is None:
        if args_dict["yamldataset"] is not None:
            args_dict["name_experiment"] = args_dict["yamldataset"].split("/")[-1].split(".")[0]
        else :
            args_dict["name_experiment"] = args_dict["dataset_name"]

        if args_dict["yamlmodel"] is not None:
            args_dict["name_experiment"] = os.path.join(args_dict["name_experiment"], args_dict["yamlmodel"].split("/")[-1].split(".")[0])

        if args_dict["yamlbatchsampler"] is not None:
            args_dict["name_experiment"] = os.path.join(args_dict["name_experiment"], args_dict["yamlbatchsampler"].split("/")[-1].split(".")[0])
        else :
            args_dict["name_experiment"] = os.path.join(args_dict["name_experiment"], args_dict["batch_sampler"])

        args_dict["name_experiment"] += time.strftime("%Y%m%d_%H%M%S")
    
    complete_weights_path = os.path.join(args_dict["model_dir"], args_dict["name_experiment"])
    if not os.path.exists(complete_weights_path):
        os.makedirs(complete_weights_path)
    args_dict["complete_weights_path"] = complete_weights_path
    args_dict["samples_dir"] = os.path.join(os.path.join(args_dict["root_dir"], "samples"),args_dict["name_experiment"])
    if not os.path.exists(args_dict["samples_dir"]):
        os.makedirs(args_dict["samples_dir"])
    args_dict["log_dir"] = os.path.join(os.path.join(args_dict["root_dir"], "logs"),args_dict["name_experiment"])
    if not os.path.exists(args_dict["log_dir"]):
        os.makedirs(args_dict["log_dir"])

    return args_dict

if __name__ == "__main__":
    args_dict = default_args()


    # Create path for saving :
    args_dict = create_path(args_dict)

    writer = SummaryWriter(log_dir=args_dict["log_dir"])
    # Add dict to summary writer
    writer.add_text("args_dict", str(args_dict))


    if args_dict["seed"] is not None:
        np.random.seed(args_dict["seed"])
        torch.manual_seed(args_dict["seed"])
        random.seed(args_dict["seed"])

    try :
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # device = torch.device('mps')
        else :
            device = torch.device('cpu')
    except AttributeError:
        device = torch.device('cpu')
    print("Working on device {}".format(device))

    # Get dataset
    complete_dataset, complete_masked_dataset = get_dataset(args_dict=args_dict)
    dataset_train = complete_masked_dataset.dataset_train
    dataset_val = complete_masked_dataset.dataset_val
    dataset_test = complete_masked_dataset.dataset_test

   
                
    example_input = dataset_train.__getitem__(0)['data']
    nb_channel = example_input.shape[0]
    args_dict['input_size'] = example_input.shape

    # Get networks :
    
    encoder_network, decoder_network, decoder_mask_network, = get_networks(args_dict=args_dict)
    encoder = get_encoder(args_dict=args_dict, encoder_network=encoder_network)
    decoder = get_decoder(args_dict=args_dict, decoder_network=decoder_network, )
    if args_dict["model_masking_process"]:
        assert decoder_mask_network is not None, "You need to specify a decoder_mask_network if you want to model masking process"
        decoder_mask = get_decoder_mask(args_dict=args_dict, decoder_mask_network=decoder_mask_network, )
    else:
        decoder_mask = None
    miwae = get_miwae(args_dict=args_dict, encoder=encoder, decoder=decoder, decoder_mask=decoder_mask)
    miwae = miwae.to(device)

           

    onepass = Onepass(model = miwae, iwae_z=args_dict["iwae_z"], mc_z=args_dict["mc_z"])

    # Get optimizer :

    if args_dict["optimizer"] == "Adam":
        optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=args_dict["lr_encoder"])
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=args_dict["lr_decoder"])
    elif args_dict["optimizer"] == "SGD_no_momentum":
        optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=args_dict["lr_encoder"], momentum=0.0)
        optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=args_dict["lr_decoder"], momentum=0.0)
    elif args_dict["optimizer"] == "SGD":
        optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=args_dict["lr_encoder"])
        optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=args_dict["lr_decoder"])
    optimizer_list = [optimizer_encoder, optimizer_decoder]
    if args_dict["model_masking_process"]:
        if args_dict["optimizer"] == "Adam":
            optimizer_decoder_mask = torch.optim.Adam(decoder_mask.parameters(), lr=args_dict["lr_decoder_mask"])
        elif args_dict["optimizer"] == "SGD_no_momentum":
            optimizer_decoder_mask = torch.optim.SGD(decoder_mask.parameters(), lr=args_dict["lr_decoder_mask"], momentum=0.0)
        elif args_dict["optimizer"] == "SGD":
            optimizer_decoder_mask = torch.optim.SGD(decoder_mask.parameters(), lr=args_dict["lr_decoder_mask"])
        miwae.compile(optim_encoder=optimizer_encoder, optim_decoder=optimizer_decoder, optim_decoder_mask=optimizer_decoder_mask)
        optimizer_list.append(optimizer_decoder_mask)
    else :
        miwae.compile(optim_encoder=optimizer_encoder, optim_decoder=optimizer_decoder)


    # Get loader :
    train_loader, strata_creator = get_dataloader(dataset=dataset_train,args = args_dict, onepass = onepass,)
    val_loader, _ = get_dataloader(dataset=dataset_val,args = args_dict, onepass = onepass, strata_creator = strata_creator,)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args_dict["batch_size"], shuffle=False,
                                              drop_last=True, num_workers=4)

    analyze_index_strata_correspondence(train_loader, args_dict, writer=writer,)

   
    # Get trainer step :
    trainer_step = get_trainer_step(args_dict["batch_sampler"], args_dict["statistics_calculation"], )
    trainer_step = trainer_step(onepass=onepass, optim_list = optimizer_list, **args_dict)
    

    # Train
    best_valid_log_likelihood= -float('inf')
    eval(iteration = 0, one_pass=onepass, val_loader=test_loader, writer=writer, args=args_dict, best_valid_log_likelihood=best_valid_log_likelihood, sample = True)
    for epoch in range(args_dict["nb_epoch"]):
        train_epoch(epoch=epoch,
                    trainer_step=trainer_step,
                    train_loader=train_loader,
                    val_loader = val_loader,
                    args=args_dict,
                    writer=writer,
                    test_loader = test_loader,
                    eval_iter = 100,
                    save_image_iter = 300,
                    best_valid_log_likelihood=best_valid_log_likelihood)

    

    # Save model :
    if not os.path.exists(args_dict["model_dir"]):
        os.makedirs(args_dict["model_dir"])
    model_path = os.path.join(args_dict["model_dir"], "model.pt")
    torch.save(miwae.state_dict(), model_path)