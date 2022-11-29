import torch
import os

from default_args import default_args
from MissingDataDataset import get_dataset
from MIWAE.get_miwae import get_miwae, get_decoder, get_encoder, get_networks
from MIWAE.TrainingUtils import train_epoch, eval

def create_path(args_dict):
    args_dict["model_dir"] = os.path.join(args_dict["root_dir"], "weights")
    if not os.path.exists(args_dict["model_dir"]):
        os.makedirs(args_dict["model_dir"])

    if "name_experiment" not in args_dict.keys() or args_dict["name_experiment"] is None:
        args_dict["name_experiment"] = args_dict["dataset_name"] + "_" + args_dict["encoder_network"] + "_" + args_dict["decoder_network"] + "_" + args_dict["encoder_reparam"] + "_" + args_dict["decoder_reparam"] + "_" + args_dict["prior"] + "_" + str(args_dict["latent_dim"])
    
    complete_weights_path = os.path.join(args_dict["model_dir"], args_dict["name_experiment"])
    if not os.path.exists(complete_weights_path):
        os.makedirs(complete_weights_path)
    args_dict["complete_weights_path"] = complete_weights_path
    args_dict["samples_dir"] = os.path.join(os.path.join(args_dict["root_dir"], "samples"),args_dict["name_experiment"])
    if not os.path.exists(args_dict["samples_dir"]):
        os.makedirs(args_dict["samples_dir"])
    
    return args_dict

if __name__ == "__main__":
    args_dict = default_args()


    # Create path for saving :
    args_dict = create_path(args_dict)

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
    dataset_test = complete_masked_dataset.dataset_test

    # Get loader :
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args_dict["batch_size"], shuffle=True, drop_last=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args_dict["batch_size"], shuffle=False,
                                              drop_last=True, num_workers=4)
                
    example_input = next(iter(train_loader))['data']
    nb_channel = example_input. shape[1]

    # Get networks :
    encoder_network, decoder_network = get_networks(args_dict=args_dict)
    encoder = get_encoder(args_dict=args_dict, encoder_network=encoder_network)
    decoder = get_decoder(args_dict=args_dict, decoder_network=decoder_network, output_channel=nb_channel)
    miwae = get_miwae(args_dict=args_dict, encoder=encoder, decoder=decoder)
    miwae = miwae.to(device)

    # Get optimizer :
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=args_dict["lr_encoder"])
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=args_dict["lr_decoder"])
    miwae.compile(optim_encoder=optimizer_encoder, optim_decoder=optimizer_decoder)
    
    # Train
    best_valid_log_likelihood= -float('inf')
    eval(epoch = -1, VAE=miwae, val_loader=test_loader,  args=args_dict, best_valid_log_likelihood=best_valid_log_likelihood)
    for epoch in range(args_dict["nb_epoch"]):
        train_epoch(epoch=epoch, VAE=miwae, train_loader=train_loader, args=args_dict)
        eval(epoch = epoch, VAE=miwae, val_loader=test_loader, args=args_dict, best_valid_log_likelihood=best_valid_log_likelihood)

    
    # Test
    eval(epoch = epoch+1, VAE=miwae, test_loader=test_loader, args=args_dict)

    # Save model :
    torch.save(miwae.state_dict(), args_dict["path_save_model"])