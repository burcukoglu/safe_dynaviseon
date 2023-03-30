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

import dynaphos
from dynaphos.cortex_models import get_visual_field_coordinates_probabilistically
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from dynaphos.utils import get_data_kwargs

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

# yaml_file= '/home/burkuc/viseon_dyna/_config/exp1_feb.yaml'
# run_path= '/home/burkuc/data/v_dy/kitty_zhao_sig_fps10_sliding_rnn_nomax_readout_cons'
# cfg = reload_model_config(yaml_file,run_path)
# models = get_model_architecture(cfg)


# models_best = load_models(models, cfg, 'best')
# models, epoch, best_perf = load_models_to_resume_training(models, cfg, 'last')

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


# valset = local_datasets.KITTI_Dataset(device=cfg['device'],
#                             directory=cfg['data_directory'],
#                             mode=cfg['mode'],
#                             n_frames=cfg['sequence_length'],
#                             imsize=(128, 128),
#                             validation=True,
#                             load_preprocessed = cfg['load_preprocessed'],
#                             sliding_sequences=cfg['sliding_sequences'])
# cfg['circular_mask'] = valset._mask.to(cfg['device'])



# cfg['circular_mask'] = valset._mask.to(cfg['device'])


# for batch_idx, batch in enumerate(valloader, 1):  # range(100):
#         batch=batch.to(cfg['device'])
#         # Forward pass
#         with torch.no_grad():
#             model_output = forward(batch, models, cfg)
#             print(model_output)
#             # exit()

def gather_data(directory, list_exp_dir):
    df_train_all=pd.DataFrame()
    df_val_all=pd.DataFrame()

    dict_exps = {}
    dict_runs = {}
    for exp_no, exp_directories in enumerate(list_exp_dir):
        print('exp_no', exp_no, 'exp_directories', exp_directories)
        df_train_exp=pd.DataFrame()
        df_val_exp=pd.DataFrame()

        
        for seed, run_dir in enumerate(exp_directories):
            print('seed', seed,"run_dir", run_dir )

            df_train_sum=pd.read_csv(directory+run_dir+"/training_summary.csv")
            df_val_sum =pd.read_csv(directory+run_dir+"/validation_summary.csv")

            df_train_exp[f"recon_loss_e{exp_no}_s{seed}"]=df_train_sum["reconstruction_loss"]
            df_val_exp[f"recon_loss_e{exp_no}_s{seed}"]=df_val_sum["reconstruction_loss"]

            df_train_exp[f"reg_loss_e{exp_no}_s{seed}"]=df_train_sum["regularization_loss"]
            df_val_exp[f"reg_loss_e{exp_no}_s{seed}"]=df_val_sum["regularization_loss"]

            df_train_exp[f"total_loss_e{exp_no}_s{seed}"]=df_train_sum["total"]
            df_val_exp[f"total_loss_e{exp_no}_s{seed}"]=df_val_sum["total"]


            dict_runs[f'train_e{exp_no}_s{seed}']=df_train_sum
            dict_runs[f'val_e{exp_no}_s{seed}']=df_val_sum
        
        dic_train_relevant_columns={key: [] for key in ["recon", "reg", "total"]}
        for col in list(df_train_exp):
            print('col', col)
            if "recon_loss" in col:
                dic_train_relevant_columns["recon"].append(col)
            elif "reg_loss" in col:
                dic_train_relevant_columns["reg"].append(col)
            elif "total_loss" in col:
                dic_train_relevant_columns["total"].append(col)
        
        dic_val_relevant_columns={key: [] for key in ["recon", "reg", "total"]}
        for col in list(df_val_exp):
            if "recon_loss" in col:
                dic_val_relevant_columns["recon"].append(col)
            elif "reg_loss" in col:
                dic_val_relevant_columns["reg"].append(col)
            elif "total_loss" in col:
                dic_val_relevant_columns["total"].append(col)

        df_train_all[f"avg_recon_loss_e{exp_no}"]= df_train_exp[dic_train_relevant_columns['recon']].mean(axis=1)
        df_train_all[f"std_recon_loss_e{exp_no}"]= df_train_exp[dic_train_relevant_columns['recon']].sem(axis=1)

        df_val_all[f"avg_recon_loss_e{exp_no}"]= df_val_exp[dic_val_relevant_columns['recon']].mean(axis=1)
        df_val_all[f"std_recon_loss_e{exp_no}"]= df_val_exp[dic_val_relevant_columns['recon']].sem(axis=1)

        df_train_all[f"avg_reg_loss_e{exp_no}"]= df_train_exp[dic_train_relevant_columns['reg']].mean(axis=1)
        df_train_all[f"std_reg_loss_e{exp_no}"]= df_train_exp[dic_train_relevant_columns['reg']].sem(axis=1)

        df_val_all[f"avg_reg_loss_e{exp_no}"]= df_val_exp[dic_val_relevant_columns['reg']].mean(axis=1)
        df_val_all[f"std_reg_loss_e{exp_no}"]= df_val_exp[dic_val_relevant_columns['reg']].sem(axis=1)

        df_train_all[f"avg_total_loss_e{exp_no}"]= df_train_exp[dic_train_relevant_columns['total']].mean(axis=1)
        df_train_all[f"std_total_loss_e{exp_no}"]= df_train_exp[dic_train_relevant_columns['total']].sem(axis=1)

        df_val_all[f"avg_total_loss_e{exp_no}"]= df_val_exp[dic_val_relevant_columns['total']].mean(axis=1)
        df_val_all[f"std_total_loss_e{exp_no}"]= df_val_exp[dic_val_relevant_columns['total']].sem(axis=1)


        dict_exps[f'train_e{exp_no}']=df_train_exp
        dict_exps[f'val_e{exp_no}']=df_val_exp

    return df_train_all, df_val_all, dict_exps, dict_runs

# CNN_UNCONS=[]
# for i in range(5):
#     run="kitty_zhao_sig_fps10_nomax_out3_uncons_s"+str(i)
#     CNN_UNCONS.append(run)

# RNN_UNCONS=[]
# RNN_UNCONS.append("kitty_zhao_sig_fps10_rnn_nomax_out3_uncons")
# for i in range(1,5):
#     run="kitty_zhao_sig_fps10_sliding_rnn_nomax_out3_uncons_s"+str(i) #redo
#     RNN_UNCONS.append(run)

# CNN_CONS_ONFR2=[]
# for i in range(5):
#     run="kitty_zhao_sig_fps10_nomax_out3_allcons2onfr_coef2_s"+str(i)
#     CNN_CONS_ONFR2.append(run)

# RNN_CONS_ONFR2=[]
# for i in range(5):
#     run="kitty_zhao_sig_fps10_sliding_rnn_nomax_out3_allcons2onfr_coef2_s"+str(i)
#     RNN_CONS_ONFR2.append(run)

# CNN_CONS1_AMP=[]
# for i in range(5):
#     run="kitty_zhao_sig_fps10_nomax_2fx_consamp_s"+str(i)
#     CNN_CONS1_AMP.append(run)

# RNN_CONS1_AMP=[]
# for i in range(5):
#     run="kitty_zhao_sig_fps10_sliding_rnn_nomax_2fx_consamp_s"+str(i)
#     RNN_CONS1_AMP.append(run)


# CNN_CONS1_PW=[]
# for i in range(5):
#     run="kitty_zhao_sig_fps10_nomax_2fx_conspw_s"+str(i)
#     CNN_CONS1_PW.append(run)


# RNN_CONS1_PW=[]
# for i in range(5):
#     run="kitty_zhao_sig_fps10_sliding_rnn_nomax_2fx_conspw_s"+str(i)
#     RNN_CONS1_PW.append(run)


# CNN_CONS1_FREQ=[]
# for i in range(5):
#     run="kitty_zhao_sig_fps10_nomax_2fx_consfreq_s"+str(i)
#     CNN_CONS1_FREQ.append(run)

# RNN_CONS1_FREQ=[]
# for i in range(5):
#     run="kitty_zhao_sig_fps10_sliding_rnn_nomax_2fx_consfreq_s"+str(i)
#     RNN_CONS1_FREQ.append(run)


# exp_dir={}

# exp_dir['CNN_UNCONS']=CNN_UNCONS
# exp_dir['RNN_UNCONS']=RNN_UNCONS

# exp_dir['CNN_CONS_ONFR2']=CNN_CONS_ONFR2
# exp_dir['RNN_CONS_ONFR2']=RNN_CONS_ONFR2

# exp_dir['CNN_CONS1_AMP']=CNN_CONS1_AMP
# exp_dir['RNN_CONS1_AMP']=RNN_CONS1_AMP

# exp_dir['CNN_CONS1_PW']=CNN_CONS1_PW
# exp_dir['RNN_CONS1_PW']=RNN_CONS1_PW

# exp_dir['CNN_CONS1_FREQ']=CNN_CONS1_FREQ
# exp_dir['RNN_CONS1_FREQ']=RNN_CONS1_FREQ

# exp_dir_uncons={}
# exp_dir_cons_onfr2={}
# exp_dir_1cons_amp={}
# exp_dir_1cons_pw={}
# exp_dir_1cons_freq={}
# exp_dir_cnn={}
# exp_dir_rnn={}

# for key, value in exp_dir.items():
#     if "UNCONS" in key:
#         exp_dir_uncons[key]=value
#     elif "CONS_ONFR2" in key:
#         exp_dir_cons_onfr2[key]=value
#     elif "CONS1_AMP" in key:
#         exp_dir_1cons_amp[key]=value
#     elif "CONS1_PW" in key:
#         exp_dir_1cons_pw[key]=value
#     elif "CONS1_FREQ" in key:
#         exp_dir_1cons_freq[key]=value
#     elif "CNN" in key:
#         exp_dir_cnn[key]=value
#     elif "RNN" in key:
#         exp_dir_rnn[key]=value



# def gather_data(directory, exp_dir):
#     df_train_all=pd.DataFrame()
#     df_val_all=pd.DataFrame()

#     dict_exps = {}
#     dict_runs = {}
#     for exp_name, exp_directories in exp_dir.items():
#         print('exp_name', exp_name, 'exp_directories', exp_directories)
#         df_train_exp=pd.DataFrame()
#         df_val_exp=pd.DataFrame()

        
#         for seed, run_dir in enumerate(exp_directories):
#             print('seed', seed,"run_dir", run_dir )

#             train_path =directory+run_dir+"/training_summary.csv"
#             val_path = directory+run_dir+"/validation_summary.csv"

#             if os.path.exists(train_path) and os.path.exists(val_path):

#                 df_train_sum=pd.read_csv(train_path)
#                 df_val_sum =pd.read_csv(val_path)

#                 df_train_exp[f"recon_loss_{exp_name}_s{seed}"]=df_train_sum["reconstruction_loss"]
#                 df_val_exp[f"recon_loss_{exp_name}_s{seed}"]=df_val_sum["reconstruction_loss"]

#                 df_train_exp[f"reg_loss_{exp_name}_s{seed}"]=df_train_sum["regularization_loss"]
#                 df_val_exp[f"reg_loss_{exp_name}_s{seed}"]=df_val_sum["regularization_loss"]

#                 df_train_exp[f"total_loss_{exp_name}_s{seed}"]=df_train_sum["total"]
#                 df_val_exp[f"total_loss_{exp_name}_s{seed}"]=df_val_sum["total"]


#                 dict_runs[f'train_e{exp_name}_s{seed}']=df_train_sum
#                 dict_runs[f'val_e{exp_name}_s{seed}']=df_val_sum
        
#         dic_train_relevant_columns={key: [] for key in ["recon", "reg", "total"]}
#         for col in list(df_train_exp):
#             print('col', col)
#             if "recon_loss" in col:
#                 dic_train_relevant_columns["recon"].append(col)
#             elif "reg_loss" in col:
#                 dic_train_relevant_columns["reg"].append(col)
#             elif "total_loss" in col:
#                 dic_train_relevant_columns["total"].append(col)
        
#         dic_val_relevant_columns={key: [] for key in ["recon", "reg", "total"]}
#         for col in list(df_val_exp):
#             if "recon_loss" in col:
#                 dic_val_relevant_columns["recon"].append(col)
#             elif "reg_loss" in col:
#                 dic_val_relevant_columns["reg"].append(col)
#             elif "total_loss" in col:
#                 dic_val_relevant_columns["total"].append(col)

#         df_train_all[f"avg_recon_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['recon']].mean(axis=1)
#         df_train_all[f"std_recon_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['recon']].sem(axis=1)

#         df_val_all[f"avg_recon_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['recon']].mean(axis=1)
#         df_val_all[f"std_recon_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['recon']].sem(axis=1)

#         df_train_all[f"avg_reg_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['reg']].mean(axis=1)
#         df_train_all[f"std_reg_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['reg']].sem(axis=1)

#         df_val_all[f"avg_reg_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['reg']].mean(axis=1)
#         df_val_all[f"std_reg_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['reg']].sem(axis=1)

#         df_train_all[f"avg_total_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['total']].mean(axis=1)
#         df_train_all[f"std_total_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['total']].sem(axis=1)

#         df_val_all[f"avg_total_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['total']].mean(axis=1)
#         df_val_all[f"std_total_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['total']].sem(axis=1)


#         dict_exps[f'train_{exp_name}']=df_train_exp
#         dict_exps[f'val_{exp_name}']=df_val_exp

#     return df_train_all, df_val_all, dict_exps, dict_runs


# def plot_across_experiments_total(exp_dir, df_train_all, df_val_all, total=True, recon=False, regul=False):
#     fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,6),dpi=120)
#     for exp in exp_dir.keys():

#         x=df_train_all.index

#         if total:
#             ax1.plot(df_train_all["avg_total_loss_"+exp], label="tr_total_loss_"+exp)#, color='tab:purple')
#             ax1.fill_between(df_train_all["avg_total_loss_"+exp]-df_train_all["std_total_loss_"+exp], df_train_all["avg_total_loss_"+exp]+df_train_all["std_total_loss_"+exp], alpha=0.35, label="tr_total_loss_SEM_"+exp)#, color='tab:purple')
            
#             ax2.plot(df_val_all["avg_total_loss_"+exp], label="val_total_loss_"+exp)#, ls='--', color='tab:purple')
#             ax2.fill_between(df_val_all["avg_total_loss_"+exp]-df_val_all["std_total_loss_"+exp], df_val_all["avg_total_loss_"+exp]+df_val_all["std_total_loss_"+exp], alpha=0.35, label="val_total_loss_SEM_"+exp)#, color='tab:purple')

#         if recon:
#             ax1.plot(df_train_all["avg_recon_loss_"+exp],label="tr_recon_loss_"+exp)#, color='tab:red' )
#             ax1.fill_between(df_train_all["avg_recon_loss_"+exp]-df_train_all["std_recon_loss_"+exp], df_train_all["avg_recon_loss_"+exp]+df_train_all["std_recon_loss_"+exp], alpha=0.35, label="tr_recon_loss_SEM_"+exp)#, color='tab:red')

#             ax2.plot(df_val_all["avg_recon_loss_"+exp],label="val_recon_loss_"+exp)#, ls='--', color='tab:red')
#             ax2.fill_between(df_val_all["avg_recon_loss_"+exp]-df_val_all["std_recon_loss_"+exp], df_val_all["avg_recon_loss_"+exp]+df_val_all["std_recon_loss_"+exp], alpha=0.35, label="val_recon_loss_SEM_"+exp)#, color='tab:red')


#         if regul:
#             ax1.plot(df_train_all["avg_reg_loss_"+exp], label="tr_reg_loss_"+exp)#, color='tab:blue')
#             ax1.fill_between(df_train_all["avg_reg_loss_"+exp]-df_train_all["std_total_loss_"+exp], df_train_all["avg_reg_loss_"+exp]+df_train_all["std_total_loss_"+exp], alpha=0.35, label="tr_reg_loss_SEM_"+exp)#, color='tab:blue')
            
#             ax2.plot(df_val_all["avg_reg_loss_"+exp], label="val_reg_loss_"+exp)#, ls='--', color='tab:blue')
#             ax2.fill_between(df_val_all["avg_reg_loss_"+exp]-df_val_all["std_total_loss_"+exp], df_val_all["avg_reg_loss_"+exp]+df_val_all["std_total_loss_"+exp], alpha=0.35, label="val_reg_loss_SEM_"+exp)#, color='tab:blue')
        

#         ax1.legend()
#         ax2.legend()
        
#     plt.show()

# exp_transform_dict={}
# for exp_name, exp_directories in exp_dir.items():
#     if "CONS1_AMP" in exp_name:
#         transform_list = ["input", "phosphenes", "reconstruction", "stimulation_amplitude",  "total_charge"]
#     elif "CONS1_PW" in exp_name:
#         transform_list = ["input", "phosphenes", "reconstruction", "stimulation_pulse_width", "total_charge"]
#     elif "CONS1_FREQ" in exp_name:
#         transform_list = ["input", "phosphenes", "reconstruction", "stimulation_frequency", "total_charge"]
#     else:
#         transform_list = ["input", "phosphenes", "reconstruction", "stimulation_amplitude",  "stimulation_pulse_width",  "stimulation_frequency",  "total_charge"]
    
#     transform_dict = {}
#     for item in transform_list:
#         if os.path.exists(directory+'output_history/{item}.pickle'):
#             f = open(directory+f'output_history/{item}.pickle', 'rb')
#             transform_dict[item]=pickle5.load(f)
#     exp_transform_dict[exp_name]=transform_dict

# def gather_data(directory, exp_dir):
#     df_train_all=pd.DataFrame()
#     df_val_all=pd.DataFrame()

#     dict_exps = {}
#     dict_runs = {}
#     for exp_name, exp_directories in exp_dir.items():
#         print('exp_name', exp_name, 'exp_directories', exp_directories)
#         df_train_exp=pd.DataFrame()
#         df_val_exp=pd.DataFrame()

        
#         for seed, run_dir in enumerate(exp_directories):
#             print('seed', seed,"run_dir", run_dir )

#             train_path =directory+run_dir+"/training_summary.csv"
#             val_path = directory+run_dir+"/validation_summary.csv"

#             if os.path.exists(train_path) and os.path.exists(val_path):

#                 df_train_sum=pd.read_csv(train_path)
#                 df_val_sum =pd.read_csv(val_path)

#                 df_train_exp[f"recon_loss_{exp_name}_s{seed}"]=df_train_sum["reconstruction_loss"]
#                 df_val_exp[f"recon_loss_{exp_name}_s{seed}"]=df_val_sum["reconstruction_loss"]

#                 df_train_exp[f"reg_loss_{exp_name}_s{seed}"]=df_train_sum["regularization_loss"]
#                 df_val_exp[f"reg_loss_{exp_name}_s{seed}"]=df_val_sum["regularization_loss"]

#                 df_train_exp[f"total_loss_{exp_name}_s{seed}"]=df_train_sum["total"]
#                 df_val_exp[f"total_loss_{exp_name}_s{seed}"]=df_val_sum["total"]


#                 dict_runs[f'train_e{exp_name}_s{seed}']=df_train_sum
#                 dict_runs[f'val_e{exp_name}_s{seed}']=df_val_sum
            
#             else:
#                 print('DOES NOT EXISTS:', train_path, os.path.exists(train_path), val_path, os.path.exists(val_path))
        
#         dic_train_relevant_columns={key: [] for key in ["recon", "reg", "total"]}
#         for col in list(df_train_exp):
#             # print('col', col)
#             if "recon_loss" in col:
#                 dic_train_relevant_columns["recon"].append(col)
#             elif "reg_loss" in col:
#                 dic_train_relevant_columns["reg"].append(col)
#             elif "total_loss" in col:
#                 dic_train_relevant_columns["total"].append(col)
        
#         dic_val_relevant_columns={key: [] for key in ["recon", "reg", "total"]}
#         for col in list(df_val_exp):
#             if "recon_loss" in col:
#                 dic_val_relevant_columns["recon"].append(col)
#             elif "reg_loss" in col:
#                 dic_val_relevant_columns["reg"].append(col)
#             elif "total_loss" in col:
#                 dic_val_relevant_columns["total"].append(col)

#         df_train_all[f"avg_recon_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['recon']].mean(axis=1)
#         df_train_all[f"std_recon_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['recon']].sem(axis=1)

#         df_val_all[f"avg_recon_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['recon']].mean(axis=1)
#         df_val_all[f"std_recon_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['recon']].sem(axis=1)

#         df_train_all[f"avg_reg_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['reg']].mean(axis=1)
#         df_train_all[f"std_reg_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['reg']].sem(axis=1)

#         df_val_all[f"avg_reg_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['reg']].mean(axis=1)
#         df_val_all[f"std_reg_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['reg']].sem(axis=1)

#         df_train_all[f"avg_total_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['total']].mean(axis=1)
#         df_train_all[f"std_total_loss_{exp_name}"]= df_train_exp[dic_train_relevant_columns['total']].sem(axis=1)

#         df_val_all[f"avg_total_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['total']].mean(axis=1)
#         df_val_all[f"std_total_loss_{exp_name}"]= df_val_exp[dic_val_relevant_columns['total']].sem(axis=1)


#         dict_exps[f'train_{exp_name}']=df_train_exp
#         dict_exps[f'val_{exp_name}']=df_val_exp

#     return df_train_all, df_val_all, dict_exps, dict_runs

# def plot_within_experiment(dict_runs, dict_exps):

#     for k, key in enumerate(dict_runs):
        
#         if "train" in key:
#             train = dict_runs[key]

#         elif "val" in key:
#             val = dict_runs[key]

#         if k%2==1:
#             fig, ax = plt.subplots()
#             ax2 = ax.twinx()
#             train.plot(x="epochs", y=["total", "reconstruction_loss","regularization_loss"], ax=ax, title=f'{key.split("_")[1]}_{key.split("_")[2]} - Seed {key.split("_")[-1]}', ylim=(0,0.25), label=["total loss " + key, "recon. loss " + key, "reg. loss " + key])
#             val.plot(x="epochs", y=["total", "reconstruction_loss","regularization_loss"], ax=ax2, ls="--", ylim=(0,0.25), label=["total loss " + key, "recon. loss " + key , "reg. loss " + key])

#             # ax.legend()
#             # ax2.legend() 
#             fig.legend()
            
#             plt.show()


# def plot_within_experiment2(dict_runs, dict_exps):


#     for k, key in enumerate(dict_runs):
        
#         if "train" in key:
#             train = dict_runs[key]

#         elif "val" in key:
#             val = dict_runs[key]

#         if k%2==1:
            
#             # fig, (ax, ax3) = plt.subplots(1,2, figsize=(20,6),dpi=120)
#             fig, ax = plt.subplots(1,1) #, figsize=(10,6),dpi=120)
#             ax2 = ax.twinx()

#             ax.plot(train['epochs'], train["total"], label="tr. total loss")
#             ax.plot(train['epochs'],train["reconstruction_loss"], label="tr. reconstruction loss")# + key.split("s")[-1])
#             ax.plot(train['epochs'], train["regularization_loss"], label="tr. regularizations loss")# + key.split("s")[-1])

#             ax2.plot(val['epochs'],val["total"], ls="--", label="val. total loss")#+ key.split("s")[-1])
#             ax2.plot(val['epochs'],val["reconstruction_loss"], ls="--", label="val. reconstruction loss")# + key.split("s")[-1])
#             ax2.plot(val['epochs'],val["regularization_loss"], ls="--", label="val. regularization loss")# + key.split("s")[-1])

#             #FOR NORMALIZING REG LOSS 
#             # ax4 = ax3.twinx()
#             # print(train["regularization_loss"].min(), train["regularization_loss"].max())
#             # print(val["regularization_loss"].min(), val["regularization_loss"].max())
#             # reg_loss_train_normalized = (train["regularization_loss"]-train["regularization_loss"].min())/(train["regularization_loss"].max()-train["regularization_loss"].min())
#             # reg_loss_val_normalized = (val["regularization_loss"]-val["regularization_loss"].min())/(val["regularization_loss"].max()-val["regularization_loss"].min())
#             # ax3.plot(train["epochs"], reg_loss_train_normalized, label="tr. reg. loss " + key.split("s")[-1])
#             # ax4.plot(val["epochs"], reg_loss_val_normalized, ls="--", label="val. reg. loss " + key.split("s")[-1])
            

#             fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#             plt.xlabel('epochs')
#             plt.title(f'{key.split("_")[1]}_{key.split("_")[2]} - Seed {key.split("_")[-1]}')
            

#             plt.show()

# def plot_across_experiment_seeds(dict_runs, dict_exps, exp_name):

#     train_total=[]
#     val_total=[]
#     train_recon=[]
#     val_recon=[]
#     train_reg=[]
#     val_reg=[]

#     for exp_key, exp_data_dict in dict_exps.items():
        

#         if exp_name in exp_key:
#             print(exp_key)
            
        
#             if "train" in exp_key:
#                 train = exp_data_dict

#                 for key, value in train.items():
#                     last_value=value.fillna(method='ffill').iloc[-1]
#                     print(key, last_value)
#                     if "total" in key:
#                         train_total.append(last_value)
#                     elif "recon" in key:
#                         train_recon.append(last_value)
#                     elif "recon" in key:
#                         train_reg.append(last_value)
                    

#             elif "val" in exp_key:
#                 val = exp_data_dict

#                 for key, value in val.items():
#                     last_value=value.fillna(method='ffill').iloc[-1]
#                     print(key, last_value)
#                     if "total" in key:
#                         val_total.append(last_value)
#                     elif "recon" in key:
#                         val_recon.append(last_value)
#                     elif "recon" in key:
#                         val_reg.append(last_value)


#     train_total=pd.DataFrame(train_total)
#     val_total=pd.DataFrame(val_total)
#     train_recon=pd.DataFrame(train_recon)
#     val_recon=pd.DataFrame(val_recon)
#     train_reg=pd.DataFrame(train_reg)
#     val_reg=pd.DataFrame(val_reg)
 
#     # return data
#     return train_total, val_total, train_recon, val_recon , train_reg, val_reg

# data_cnn_uncons=plot_across_experiment_seeds(dict_runs, dict_exps, "CNN_UNCONS")

# #within exp loss values
# fig = plt.figure(figsize =(10, 7))

# df=[]
# cnn_uncons_tr_total=pd.DataFrame({'Experiment':['CNN_UNCONS 0', 'CNN_UNCONS 1', 'CNN_UNCONS 3', 'CNN_UNCONS 4'], 'Tr. Total Loss':data_cnn_uncons[0][0]})
# cnn_uncons_tr_recon=pd.DataFrame({'Experiment':['CNN_UNCONS 0', 'CNN_UNCONS 1', 'CNN_UNCONS 3', 'CNN_UNCONS 4'], 'Tr. Recon. Loss':data_cnn_uncons[2][0]})
# # cnn_uncons_tr_reg=pd.DataFrame({'Experiment':['CNN_UNCONS 0', 'CNN_UNCONS 1', 'CNN_UNCONS 3', 'CNN_UNCONS 4'], 'Tr. Reg. Loss':data_cnn_uncons[4][0]})
# # canny_avg=pd.DataFrame({'Experiment':['Canny 0', 'Canny 1', 'Canny 2', 'Canny 3', 'Canny 4'], 'Avg.':canny_reward_avg.fillna(method='ffill').iloc[-1]})
# # e2e_avg=pd.DataFrame({'Experiment':['E2E 0', 'E2E 1', 'E2E 2', 'E2E 3', 'E2E 4'], 'Avg.':e2e_reward_avg.fillna(method='ffill').iloc[-1]})
# # blind_avg=pd.DataFrame({'Experiment':['Blind 0', 'Blind 1', 'Blind 2', 'Blind 3', 'Blind 4'], 'Avg.':blind_reward_avg.fillna(method='ffill').iloc[-1]})

# # print(normal_avg)
# df_tr=pd.concat([cnn_uncons_tr_total,cnn_uncons_tr_recon])#,cnn_uncons_tr_reg])#, canny_avg, e2e_avg, blind_avg])


# cnn_uncons_val_total=pd.DataFrame({'Experiment':['CNN_UNCONS 0', 'CNN_UNCONS 1', 'CNN_UNCONS 3', 'CNN_UNCONS 4'], 'Val. Total Loss':data_cnn_uncons[1][0]})
# cnn_uncons_val_recon=pd.DataFrame({'Experiment':['CNN_UNCONS 0', 'CNN_UNCONS 1', 'CNN_UNCONS 3', 'CNN_UNCONS 4'], 'Val. Recon. Loss':data_cnn_uncons[3][0]})
# # cnn_uncons_val_reg=pd.DataFrame({'Experiment':['CNN_UNCONS 0', 'CNN_UNCONS 1', 'CNN_UNCONS 3', 'CNN_UNCONS 4'], 'Val. Reg Loss':data_cnn_uncons[5][0]})

# # canny_max=pd.DataFrame({'Experiment':['Canny 0', 'Canny 1', 'Canny 2', 'Canny 3', 'Canny 4'], 'Max.':canny_reward_max.fillna(method='ffill').iloc[-1]})
# # e2e_max=pd.DataFrame({'Experiment':['E2E 0', 'E2E 1', 'E2E 2', 'E2E 3', 'E2E 4'], 'Max.':e2e_reward_max.fillna(method='ffill').iloc[-1]})
# # blind_max=pd.DataFrame({'Experiment':['Blind 0', 'Blind 1', 'Blind 2', 'Blind 3', 'Blind 4'], 'Max.':blind_reward_max.fillna(method='ffill').iloc[-1]})

# df_val=pd.concat([cnn_uncons_val_total,cnn_uncons_val_recon])#,cnn_uncons_val_reg])#, canny_max, e2e_max, blind_max])


# df=df_tr.join(df_val.set_index('Experiment'), on='Experiment')
# df['Experiment']=df['Experiment'].apply(lambda x: x.split(' ')[0])

# print(df)


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# dd=pd.melt(df,id_vars=['Experiment'],value_vars=['Tr. Total Loss','Val. Total Loss'],var_name='Loss')
# sns.set_theme()#style="whitegrid")
# sns.set_context("paper", font_scale=2.5) 
# sns.boxplot(x='Experiment', y='value',data=dd,hue='Loss', width=0.6)
# plt.ylabel('')
# plt.xlabel('')
# plt.yticks(fontsize=18)
# plt.xticks(fontsize=18)
# plt.title('Breakout', fontsize=26)
# plt.legend()

# plt.legend(fontsize=19, bbox_to_anchor=(1.15, 1.0),loc='upper right')
# plt.tight_layout()

# plt.show()


# fig = plt.figure(figsize =(25, 7))
 
# # Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])
 

# # normal=normalv_reward_max.fillna(method='ffill').iloc[-1]
# # canny=canny_reward_max.fillna(method='ffill').iloc[-1]
# # e2e=e2e_reward_max.fillna(method='ffill').iloc[-1]
# # blind=blind_reward_max.fillna(method='ffill').iloc[-1]

# data2=[data_cnn_uncons[0][0], 
#         data_rnn_uncons[0][0],
#         data_cnn_cons_onfr2[0][0], 
#         data_rnn_cons_onfr2[0][0], 
#         # data_cnn_cons1_amp[0][0],
#         data_rnn_cons1_amp[0][0],
#         data_cnn_cons1_pw[0][0], 
#         # data_rnn_cons1_pw[0][0],
#         data_cnn_cons1_freq[0][0], 
#         data_rnn_cons1_freq[0][0]]
# # Creating plot
# bp = ax.boxplot(data2)
# ax.set_xticklabels(['CNN Uncons.', 
#                     'RNN Uncons.',
#                     'CNN All Cons.', 
#                     'RNN All Cons.',
#                     # 'CNN Ampl. Cons.', 
#                     'RNN Ampl. Cons.',  
#                     'CNN PW Cons.', 
#                     # 'RNN PW Cons.', 
#                     'CNN Freq. Cons.', 
#                     'RNN Freq. Cons.'])
# ax.set_title('Final Training Loss')
# # show plot
# # plt.savefig('breakout_box_maxrew')
# plt.show()


# fig = plt.figure(figsize =(25, 7))
 
# # Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])
 

# data2=[data_cnn_uncons[1][0], 
#         data_rnn_uncons[1][0], 
#         data_cnn_cons_onfr2[1][0], 
#         data_rnn_cons_onfr2[1][0], 
#         # data_cnn_cons1_amp[0][0], 
#         data_rnn_cons1_amp[1][0], 
#         data_cnn_cons1_pw[1][0], 
#         # data_rnn_cons1_pw[0][0], 
#         data_cnn_cons1_freq[1][0],
#         data_rnn_cons1_freq[1][0]]
# # Creating plot
# bp = ax.boxplot(data2)
# ax.set_xticklabels(['CNN Uncons.', 
#                     'RNN Uncons.',
#                     'CNN All Cons.', 
#                     'RNN All Cons.',
#                     # 'CNN Ampl. Cons.', 
#                     'RNN Ampl. Cons.',  
#                     'CNN PW Cons.', 
#                     # 'RNN PW Cons.', 
#                     'CNN Freq. Cons.', 
#                     'RNN Freq. Cons.'])
# ax.set_title('Final Validation Loss')
# # show plot
# # plt.savefig('breakout_box_maxrew')
# plt.show()

# def boxplots_over_seeds(loss_to_plot):
#     fig = plt.figure(figsize =(25, 7))
    
#     # Creating axes instance
#     ax = fig.add_axes([0, 0, 1, 1])

#     if loss_to_plot=="train_total":
#         i=0
#         loss_type='Total'
#     elif loss_to_plot=="val_total":
#         i=1
#         loss_type='Total'
#     elif loss_to_plot=="train_recon":
#         i=2
#         loss_type='Reconstruction'
#     elif loss_to_plot=="val_recon":
#         i=3
#         loss_type='Reconstruction'
#     elif loss_to_plot=="train_reg":
#         i=4
#         loss_type='Regularization'
#     elif loss_to_plot=="val_reg":
#         i=5
#         loss_type='Regularization'

#     if i%2==0:
#         mode='Training'
#     else:
#         mode='Validation'


#     data2=[
#             data_cnn_uncons[i][0], 
#             data_rnn_uncons[i][0], 
#             data_cnn_cons_onfr2[i][0], 
#             data_rnn_cons_onfr2[i][0], 

#             # data_cnn_cons1_amp[i][0], 

#             data_rnn_cons1_amp[i][0], 
#             data_cnn_cons1_pw[i][0], 

#             # data_rnn_cons1_pw[i][0], 

#             data_cnn_cons1_freq[i][0],
#             data_rnn_cons1_freq[i][0]
#             ]
#     # Creating plot
#     bp = ax.boxplot(data2)
#     ax.set_xticklabels([
#                         'CNN Uncons.', 
#                         'RNN Uncons.',
#                         'CNN All Cons.', 
#                         'RNN All Cons.',

#                         # 'CNN Ampl. Cons.', 

#                         'RNN Ampl. Cons.',  
#                         'CNN PW Cons.', 

#                         # 'RNN PW Cons.', 

#                         'CNN Freq. Cons.', 
#                         'RNN Freq. Cons.'
#                         ])
#     ax.set_title(f'Final {mode} {loss_type} Loss')
#     # show plot
#     # plt.savefig('breakout_box_maxrew')
#     plt.show()

# boxplots_over_seeds(loss_to_plot='train_total')