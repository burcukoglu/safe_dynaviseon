import matplotlib.pyplot as plt
import pandas as pd
import pickle5 as pickle

import numpy as np
from PIL import Image
import cv2
import imageio
import yaml
import torch
import os


import tensorboard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

import model
import init_training

def load_transform(directory): #out1
    transform_list = ["input", "phosphenes", "reconstruction", "stimulation"]
    transform_dict = {}
    for item in transform_list:
        f = open(directory+f'/output_history/{item}.pickle', 'rb')
        transform_dict[item]=pickle.load(f)
        # print(transform_dict[item])
    return transform_dict

def load_transform2(directory): #out3
    transform_list = ["input", "phosphenes", "reconstruction", "stimulation_amplitude",  "stimulation_frequency",  "stimulation_pulse_width"]
    transform_dict = {}
    for item in transform_list:
        f = open(directory+f'/output_history/{item}.pickle', 'rb')
        transform_dict[item]=pickle.load(f)
        # print(transform_dict[item])
    return transform_dict

def prepare_gif(transform_dict, title):

    if len(transform_dict["input"][0].shape)==5:
        seq_length = transform_dict["input"][0].shape[2]
    elif len(transform_dict["input"][0].shape)==4:
        seq_length = transform_dict["input"][0].shape[1]
        
    fig, axes = plt.subplots(nrows=len(transform_dict.keys())-3, ncols=seq_length, figsize=(20,6),dpi=120)

    img_typ=len(list(transform_dict.values()))

    image_list=[]

    for i, ax in enumerate(axes.flatten()):
        

        k = i//seq_length

        j = i%seq_length
        # print('i:', i, ' k:', k, ' j:', j)

        if len(list(transform_dict.values())[k][0].shape)>3:
            if len(list(transform_dict.values())[k][0].shape) == 4:
                image = list(transform_dict.values())[k][-1][0,j,:,:]



            elif len(list(transform_dict.values())[k][0].shape) == 5:
                image = list(transform_dict.values())[k][-1][0,0,j,:,:]

            
        image_list.append(image)

        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.grid(visible=False)
        
    image_array=np.stack(image_list[:seq_length], axis=0)
    phos_array=np.stack(image_list[seq_length:seq_length*2], axis=0)
    recon_array=np.stack(image_list[seq_length*2:], axis=0)

        
    fig.suptitle(title, fontsize=16)
    plt.show()
    fig.savefig(f'/home/burkuc/data/v_dy/figs/{title}.png', dpi=fig.dpi)
    return image_array, phos_array, recon_array

def generate_video(images_for_gif, phosphenes_for_gif, recon_for_gif, title):
    frames_for_gif =np.zeros((phosphenes_for_gif.shape[0],phosphenes_for_gif.shape[1],phosphenes_for_gif.shape[2]*3))
    for idx, (frame_idx, phos_idx, recon_idx)  in enumerate(zip(images_for_gif,phosphenes_for_gif,recon_for_gif)):
        frame_idx = cv2.resize(frame_idx, (256,256), interpolation=cv2.INTER_LINEAR)
        recon_idx = cv2.resize(recon_idx, (256,256), interpolation=cv2.INTER_LINEAR)
        frames_for_gif[idx] = np.concatenate((frame_idx,phos_idx, recon_idx), axis=-1)

    imageio.mimsave(f"/home/burkuc/data/v_dy/gifs/{title}.mp4",
                    frames_for_gif, fps=10)

def generate_gif(images_for_gif, phosphenes_for_gif, recon_for_gif, title):
    frames_for_gif =np.zeros((phosphenes_for_gif.shape[0],phosphenes_for_gif.shape[1],phosphenes_for_gif.shape[2]*3))
    for idx, (frame_idx, phos_idx, recon_idx)  in enumerate(zip(images_for_gif,phosphenes_for_gif,recon_for_gif)):
        frame_idx = cv2.resize(frame_idx, (256,256), interpolation=cv2.INTER_LINEAR)
        recon_idx = cv2.resize(recon_idx, (256,256), interpolation=cv2.INTER_LINEAR)
        frames_for_gif[idx] = np.concatenate((frame_idx,phos_idx, recon_idx), axis=-1)

    imageio.mimsave(f"/home/burkuc/data/v_dy/gifs/{title}.gif",
                    frames_for_gif, format = 'GIF-PIL', duration=1/10)

def reload_model_config(yaml_file, run_path):
    with open(yaml_file) as file:
        raw_content = yaml.load(file,Loader=yaml.FullLoader) # nested dictionary
    cfg= {k:v for params in raw_content.values() for k,v in params.items()} # unpacked
    cfg['device'] ='cpu'
    cfg['gpu'] =None
    cfg['save_path'] = run_path
    return cfg
    
def get_encoder_architecture(cfg):
    if cfg['model_architecture'] == 'recurrent_net':
        encoder, decoder = model.get_e2e_recurrent_net(cfg)
    #added
    elif cfg['model_architecture'] == 'recurrent_net_out3':
        encoder, decoder = model.get_e2e_recurrent_net_out3(cfg)
    #added
    elif cfg['model_architecture'] == 'recurrent_net_out32':
        encoder, decoder = model.get_e2e_recurrent_net_out32(cfg)
    elif cfg['model_architecture'] == 'zhao-autoencoder':
        encoder, decoder = model.get_Zhao_autoencoder(cfg)
    #added
    elif cfg['model_architecture'] == 'zhao-autoencoder_out3':
        encoder, decoder = model.get_Zhao_autoencoder_out3(cfg)
    #added
    elif cfg['model_architecture'] == 'zhao-autoencoder_out32':
        encoder, decoder = model.get_Zhao_autoencoder_out32(cfg)
    else:
        raise NotImplementedError
    return encoder

def load_params(model, cfg, prefix):
    fn = os.path.join(cfg['save_path'], 'checkpoints', f'{prefix}_encoder.pth')
    model.load_state_dict(torch.load(fn, map_location=cfg['device']))
    return model

def load_checkpoint_with_cfg(model, cfg, prefix='last'):
    fn = os.path.join(cfg['save_path'], 'checkpoints', f'{prefix}_checkpoint.pth')
    state = torch.load(fn, map_location=cfg['device'])
    model.load_state_dict(state['state_dict_encoder'])
    epoch = state["epoch"]
    cfg = state["cfg"] 
    best_validation_performance = state["best_validation_performance"]
    # return epoch, batch_idx, best_validation_performance
    return model, cfg, epoch, best_validation_performance

# def load_checkpoint_from_path(model, save_path, prefix='last'):
#     fn = os.path.join(save_path, 'checkpoints', f'{prefix}_checkpoint.pth')
#     state = torch.load(fn, map_location='cpu')
#     model.load_state_dict(state['state_dict_encoder'])
#     epoch = state["epoch"]
#     cfg = state["cfg"] 
#     best_validation_performance = state["best_validation_performance"]
#     return model, cfg, epoch, best_validation_performance

def get_training_pipeline(cfg):
    if cfg['pipeline'] == 'unconstrained-video-reconstruction':
        forward, lossfunc = init_training.get_pipeline_unconstrained_video_reconstruction(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_rnn':
        forward, lossfunc = init_training.get_pipeline_unconstrained_video_reconstruction_rnn(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_rnn_out3':
        forward, lossfunc = init_training.get_pipeline_unconstrained_video_reconstruction_rnn_out3(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_out3':
        forward, lossfunc = init_training.get_pipeline_unconstrained_video_reconstruction_out3(cfg)
    else:
        print(cfg['pipeline'] + 'not supported yet')
        raise NotImplementedError

    return {'forward': forward, 'compound_loss_func': lossfunc}

#FINAL VERS
def reload_model_config(yaml_file, run_path):
    with open(yaml_file) as file:
        raw_content = yaml.load(file,Loader=yaml.FullLoader) # nested dictionary
    cfg= {k:v for params in raw_content.values() for k,v in params.items()} # unpacked
    # cfg['device'] = 'cpu'
    # cfg['gpu'] = None
    cfg['save_path'] = run_path
    return cfg

def get_simulator(cfg):
    # initialise simulator
    params = dynaphos.utils.load_params(cfg['base_config'])
    params['run'].update(cfg)
    params['thresholding'].update(cfg)
    device = get_data_kwargs(params)['device']

    with open(cfg['phosphene_map'], 'rb') as handle:
        coordinates_visual_field = pickle.load(handle, )
    simulator = PhospheneSimulator(params, coordinates_visual_field)
    cfg['SPVsize'] = simulator.phosphene_maps.shape[-2:]
    return simulator

def get_model_architecture(cfg):
    if cfg['model_architecture'] == 'recurrent_net':
        encoder, decoder = m.get_e2e_recurrent_net(cfg)
    #added
    elif cfg['model_architecture'] == 'recurrent_net_out3':
        encoder, decoder = m.get_e2e_recurrent_net_out3(cfg)
    #added
    elif cfg['model_architecture'] == 'recurrent_net_out32':
        encoder, decoder = m.get_e2e_recurrent_net_out32(cfg)
    elif cfg['model_architecture'] == 'zhao-autoencoder':
        encoder, decoder = m.get_Zhao_autoencoder(cfg)
    #added
    elif cfg['model_architecture'] == 'zhao-autoencoder_out3':
        encoder, decoder = m.get_Zhao_autoencoder_out3(cfg)
    #added
    elif cfg['model_architecture'] == 'zhao-autoencoder_out32':
        encoder, decoder = m.get_Zhao_autoencoder_out32(cfg)
    else:
        raise NotImplementedError

    simulator = get_simulator(cfg)
    # print('simul', simulator.device)

    models = {'encoder' : encoder,
              'decoder' : decoder,
            #   'optimizer': optimizer,
              'simulator': simulator,}

    return models

def load_models(models, cfg, prefix='best'):
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            fn = os.path.join(cfg['save_path'], 'checkpoints', f'{prefix}_{name}.pth')
            model.load_state_dict(torch.load(fn, map_location=cfg['device']))
    return models

def load_models_to_resume_training(models, cfg, prefix='last'):
    fn = os.path.join(cfg['save_path'], 'checkpoints', f'{prefix}_checkpoint.pth')
    state = torch.load(fn, map_location=cfg['device'])
    for name, model in models.items():
        # print('name, model', name, model)
        if isinstance(model, torch.nn.Module):
            # print('isinstance name, model', name, model)
            model.load_state_dict(state['state_dict_'+name])
    epoch = state["epoch"]
    # batch_idx = state["batch_idx"]
    cfg = state["cfg"] 
    best_validation_performance = state["best_validation_performance"]
    # return epoch, batch_idx, best_validation_performance
    return models, epoch, best_validation_performance

yaml_file= '/home/burkuc/viseon_dyna/_config/exp1_feb.yaml'
run_path= '/home/burkuc/data/v_dy/kitty_zhao_sig_fps10_sliding_rnn_nomax_readout_cons'
cfg = reload_model_config(yaml_file,run_path)
models = get_model_architecture(cfg)


models_best = load_models(models, cfg, 'best')
models, epoch, best_perf = load_models_to_resume_training(models, cfg, 'last')

def get_training_pipeline(cfg):
    if cfg['pipeline'] == 'unconstrained-video-reconstruction':
        forward, lossfunc = init_training.get_pipeline_unconstrained_video_reconstruction(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_rnn':
        forward, lossfunc = init_training.get_pipeline_unconstrained_video_reconstruction_rnn(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_rnn_out3':
        forward, lossfunc = init_training.get_pipeline_unconstrained_video_reconstruction_rnn_out3(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_out3':
        forward, lossfunc = init_training.get_pipeline_unconstrained_video_reconstruction_out3(cfg)
    else:
        print(cfg['pipeline'] + 'not supported yet')
        raise NotImplementedError

    return {'forward': forward, 'compound_loss_func': lossfunc}


valset = local_datasets.KITTI_Dataset(device=cfg['device'],
                            directory=cfg['data_directory'],
                            mode=cfg['mode'],
                            n_frames=cfg['sequence_length'],
                            imsize=(128, 128),
                            validation=True,
                            load_preprocessed = cfg['load_preprocessed'],
                            sliding_sequences=cfg['sliding_sequences'])
# cfg['circular_mask'] = valset._mask.to(cfg['device'])



cfg['circular_mask'] = valset._mask.to(cfg['device'])


for batch_idx, batch in enumerate(valloader, 1):  # range(100):
        batch=batch.to(cfg['device'])
        # Forward pass
        with torch.no_grad():
            model_output = forward(batch, models, cfg)
            print(model_output)
            # exit()