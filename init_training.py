import os
import torch
import numpy as np
import pickle

#added
import sys
sys.path.insert(1, '/home/burkuc/dynaphos')
import pdb
import matplotlib.pyplot as plt

import dynaphos
from dynaphos.cortex_models import get_visual_field_coordinates_probabilistically
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from dynaphos.utils import get_data_kwargs

import model

import local_datasets
from torch.utils.data import DataLoader
from utils import resize, normalize, dilation3x3, CustomSummaryTracker

from torch.utils.tensorboard import SummaryWriter


class LossTerm():
    """Loss term that can be used for the compound loss"""

    def __init__(self, name=None, func=torch.nn.functional.mse_loss, arg_names=None, weight=1.):
        self.name = name
        self.func = func  # the loss function
        self.arg_names = arg_names  # the names of the inputs to the loss function
        self.weight = weight  # the relative weight of the loss term


class CompoundLoss():
    """Helper class for combining multiple loss terms. Initialize with list of
    LossTerm instances. Returns dict with loss terms and total loss"""

    def __init__(self, loss_terms):
        self.loss_terms = loss_terms

    def __call__(self, loss_targets):
        """Calculate all loss terms and the weighted sum"""
        self.out = dict()
        self.out['total'] = 0
        for lt in self.loss_terms:
            func_args = [loss_targets[name] for name in lt.arg_names]  # Find the loss targets by their name
            self.out[lt.name] = lt.func(*func_args)  # calculate result and add to output dict
            self.out['total'] += self.out[lt.name] * lt.weight  # add the weighted loss term to the total
        return self.out

    def items(self):
        """return dict with loss tensors as dict with Python scalars"""
        return {k: v.item() for k, v in self.out.items()}


class RunningLoss():
    """Helper class to track the running loss over multiple batches."""

    def __init__(self):
        self.dict = dict()
        self.reset()

    def reset(self):
        self._counter = 0
        for key in self.dict.keys():
            self.dict[key] = 0.

    def update(self, new_entries):
        """Add the current loss values to the running loss"""
        self._counter += 1
        for key, value in new_entries.items():
            if key in self.dict:
                self.dict[key] += value
            else:
                self.dict[key] = value

    def get(self):
        """Get the average loss values (total loss dived by the processed batch count)"""
        out = {key: (value / self._counter) for key, value in self.dict.items()}
        return out


class L1FeatureLoss(object):
    def __init__(self):
        self.feature_extractor = model.VGGFeatureExtractor(device=device)
        self.loss_fn = torch.nn.functional.l1_loss

    def __call__(self, y_pred, y_true, ):
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        err = [self.loss_fn(pred, true) for pred, true in zip(pred_features, true_features)]
        return torch.mean(torch.stack(err))



def get_dataset(cfg):
    if cfg['dataset'] == 'ADE50K':
        trainset, valset = local_datasets.get_ade50k_dataset(cfg)
    elif cfg['dataset'] == 'BouncingMNIST':
        trainset, valset = local_datasets.get_bouncing_mnist_dataset(cfg)
    elif cfg['dataset'] == 'KITTI':
        trainset, valset = local_datasets.get_kitti_dataset(cfg)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],shuffle=True, drop_last=True)
    valloader = DataLoader(valset,batch_size=cfg['batch_size'],shuffle=False, drop_last=True)
    example_batch = next(iter(valloader))
    cfg['circular_mask'] = trainset._mask.to(cfg['device'])

    dataset = {'trainset': trainset,
               'valset': valset,
               'trainloader': trainloader,
               'valloader': valloader,
               'example_batch': example_batch}

    return dataset


def get_models(cfg):
    if cfg['model_architecture'] == 'end-to-end-autoencoder':
        encoder, decoder = model.get_e2e_autoencoder(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    #added
    elif cfg['model_architecture'] == 'recurrent_net':
        encoder, decoder = model.get_e2e_recurrent_net(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    #added
    elif cfg['model_architecture'] == 'recurrent_net_out3':
        encoder, decoder = model.get_e2e_recurrent_net_out3(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    #added
    elif cfg['model_architecture'] == 'recurrent_net_out32':
        encoder, decoder = model.get_e2e_recurrent_net_out32(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    #added
    elif cfg['model_architecture'] == 'end-to-end-autoencoder2':
        encoder, decoder = model.get_e2e_autoencoder2(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    elif cfg['model_architecture'] == 'zhao-autoencoder':
        encoder, decoder = model.get_Zhao_autoencoder(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    #added
    elif cfg['model_architecture'] == 'zhao-autoencoder2':
        encoder, decoder = model.get_Zhao_autoencoder2(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    #added
    elif cfg['model_architecture'] == 'zhao-autoencoder_out3':
        encoder, decoder = model.get_Zhao_autoencoder_out3(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    #added
    elif cfg['model_architecture'] == 'zhao-autoencoder_out32':
        encoder, decoder = model.get_Zhao_autoencoder_out32(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    else:
        raise NotImplementedError

    simulator = get_simulator(cfg)
    # print('simul', simulator.device)

    models = {'encoder' : encoder,
              'decoder' : decoder,
              'optimizer': optimizer,
              'simulator': simulator,}

    return models


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


def get_logging(cfg):
    out = dict()
    out['training_loss'] = RunningLoss()
    out['validation_loss'] = RunningLoss()
    out['tensorboard_writer'] = SummaryWriter(os.path.join(cfg['save_path'], 'tensorboard/'))
    out['training_summary'] = CustomSummaryTracker()
    out['validation_summary'] = CustomSummaryTracker()
    out['example_output'] = CustomSummaryTracker()
    return out

####### ADJUST OR ADD TRAINING PIPELINE BELOW

def get_training_pipeline(cfg):
    if cfg['pipeline'] == 'supervised-boundary-reconstruction':
        forward, lossfunc = get_pipeline_supervised_boundary_reconstruction(cfg)
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_rnn':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction_rnn(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_rnn_out3':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction_rnn_out3(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction2':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction2(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction3':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction3(cfg)
    #added
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_out3':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction_out3(cfg)
    else:
        print(cfg['pipeline'] + 'not supported yet')
        raise NotImplementedError

    return {'forward': forward, 'compound_loss_func': lossfunc}


def get_pipeline_supervised_boundary_reconstruction(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, label = batch
        print('image , label', image.shape, label.shape)
        label = dilation3x3(label)
        print('dilated label',  label.shape)

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        print('stim', stimulation.shape)
        phosphenes = simulator(stimulation).unsqueeze(1)
        print('phosphenes', phosphenes.shape)
        reconstruction = decoder(phosphenes) * cfg['circular_mask']
        print('recon', reconstruction.shape)
        

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction * cfg['circular_mask'],
               'target': label * cfg['circular_mask'],
               'target_resized': resize(label * cfg['circular_mask'], cfg['SPVsize'],),}

        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes) ,
                    'target_centers': simulator.sample_centers(out['target_resized']) })

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'target'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_centers', 'target_centers'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func



def get_pipeline_unconstrained_video_reconstruction(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        # Unpack
        # print('batch', len(batch), batch[0].shape, batch[1].shape) 
        # frames = batch
        #added/changed
        if isinstance(batch, list):
            frames = batch[0]
            # print('islist')
        else:
            frames = batch

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # print('simul', simulator.device)
        # bouncing , mode: recon >> batch dtype torch.float32 batch len 2 but its a tensor of torch.Size([2, 1, 5, 128, 128])  not list, so elements:  torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        # bouncing , mode: recon_pred or ade20k image,label >> batch 2 but it's a list of 2 elements(image, label or current,future), so elements:  torch.Size([2, 5, 128, 128]) torch.Size([2, 5, 128, 128])
        
        # print('frames', frames.shape, len(frames), frames[0].shape, frames[1].shape) # torch.Size([2, 1, 5, 128, 128]) 2 torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        
        # pdb.set_trace()
        # pdb.enable()

        # Forward
        simulator.reset()
        stimulation_sequence = encoder(frames).permute(1, 0, 2)  # permute: (Batch,Time,Num_phos) -> (Time,Batch,Num_phos) #
        # print('stimulation_sequence reshaped', stimulation_sequence.shape) #torch.Size([5, 2, 1000])
        # print('stimulation_sequence', stimulation_sequence.min(), stimulation_sequence.max())
        phosphenes = []
        # for stim in stimulation_sequence:
        #     phosphenes.append(simulator(stim))  # simulator expects (Batch, Num_phosphenes)
        # phosphenes = torch.stack(phosphenes, dim=1).unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)

        #changed/added
        total_charges = []
        for stim in stimulation_sequence:
            phos, total_charge= simulator(stim)
            # phosphenes.append(simulator(amplitude, pulse_width, frequency))
            phosphenes.append(phos)
            total_charges.append(total_charge)
            # print('tc sh',total_charge.shape)


        phosphenes = torch.stack(phosphenes, dim=1).unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        #added
        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) #.unsqueeze(dim=1)
        # print('tc', total_charges.shape)
        # print('phosphenes', phosphenes.min(), phosphenes.max())
        # print('shape',phosphenes.shape[2])
        # for i in range(phosphenes.shape[2]):
        #     plt.imsave(f'/home/burkuc/data/static/phos_v_{i}.png', phosphenes[0,0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # plt.imsave(f'/home/burkuc/data/static/phos_v_all.png', phosphenes[0,0,:,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # print('phosphenes reshaped', phosphenes.shape) #torch.Size([2, 1, 5, 256, 256])
        reconstruction = decoder(phosphenes)
        # print('reconstruction ', reconstruction.shape) #torch.Size([2, 1, 5, 128, 128])
        # pdb.set_trace()
        # pdb.disable()

        out =  {'stimulation': stimulation_sequence, #torch.Size([5, 2, 1000])
                'total_charge': total_charges, #added
                'phosphenes': phosphenes,#torch.Size([2, 1, 5, 256, 256])
                'reconstruction': reconstruction * cfg['circular_mask'], ##torch.Size([2, 1, 5, 128, 128])
                'input': frames * cfg['circular_mask'], #torch.Size([2, 1, 5, 128, 128])
                'input_resized': resize(frames *cfg['circular_mask'],
                                         (cfg['sequence_length'],*cfg['SPVsize']),interpolation='trilinear')} #torch.Size([2, 1, 5, 256, 256])
        # print('phos',out['phosphenes'].min(),out['phosphenes'].max())
        # print('input_resized',out['input_resized'].min(),out['input_resized'].max())
        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}

        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight= cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

#added
def get_pipeline_unconstrained_video_reconstruction_out3(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        # Unpack
        # print('batch', len(batch), batch[0].shape, batch[1].shape) 
        # frames = batch
        #added/changed
        if isinstance(batch, list):
            frames = batch[0]
            # print('islist')
        else:
            frames = batch

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # print('simul', simulator.device)
        # bouncing , mode: recon >> batch dtype torch.float32 batch len 2 but its a tensor of torch.Size([2, 1, 5, 128, 128])  not list, so elements:  torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        # bouncing , mode: recon_pred or ade20k image,label >> batch 2 but it's a list of 2 elements(image, label or current,future), so elements:  torch.Size([2, 5, 128, 128]) torch.Size([2, 5, 128, 128])
        
        # print('frames', frames.shape, len(frames), frames[0].shape, frames[1].shape) # torch.Size([2, 1, 5, 128, 128]) 2 torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        
        # pdb.set_trace()
        # pdb.enable()

        # Forward
        simulator.reset()
        amplitude_seq, pulse_width_seq, frequency_seq = encoder(frames)

        amplitude_seq = amplitude_seq.permute(1, 0, 2)
        pulse_width_seq = pulse_width_seq.permute(1, 0, 2)
        frequency_seq = frequency_seq.permute(1, 0, 2)

        # stimulation_sequence = encoder(frames).permute(1, 0, 2)  # permute: (Batch,Time,Num_phos) -> (Time,Batch,Num_phos) #
        # print('stimulation_sequence reshaped', stimulation_sequence.shape) #torch.Size([5, 2, 1000])
        # print('stimulation_sequence', stimulation_sequence.min(), stimulation_sequence.max())
        # print('stimulation_sequence',stimulation_sequence)
        phosphenes = []
        # for stim in stimulation_sequence:
        #     phosphenes.append(simulator(stim))  # simulator expects (Batch, Num_phosphenes)

        # for amplitude, pulse_width, frequency in zip(amplitude_seq, pulse_width_seq, frequency_seq):
        #     phosphenes.append(simulator(amplitude, pulse_width, frequency))
        #changed
        total_charges = []
        for amplitude, pulse_width, frequency in zip(amplitude_seq, pulse_width_seq, frequency_seq):
            phos, total_charge= simulator(amplitude, pulse_width, frequency)
            # phosphenes.append(simulator(amplitude, pulse_width, frequency))
            phosphenes.append(phos)
            total_charges.append(total_charge)
            # print('tc sh',total_charge.shape)


        phosphenes = torch.stack(phosphenes, dim=1).unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        #added
        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) #.unsqueeze(dim=1)
        # print('tcss sh min  max',total_charges.shape) #, total_charges.min(), total_charges.max())
        # print('phosphenes', phosphenes.min(), phosphenes.max())
        # print('shape',phosphenes.shape[2])
        # for i in range(phosphenes.shape[2]):
        #     plt.imsave(f'/home/burkuc/data/static/phos_v_{i}.png', phosphenes[0,0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # plt.imsave(f'/home/burkuc/data/static/phos_v_all.png', phosphenes[0,0,:,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # print('phosphenes reshaped', phosphenes.shape) #torch.Size([2, 1, 5, 256, 256])
        reconstruction = decoder(phosphenes)
        # print('reconstruction ', reconstruction.shape) #torch.Size([2, 1, 5, 128, 128])
        # pdb.set_trace()
        # pdb.disable()

        out =  {'stimulation_amplitude': amplitude_seq,
                'stimulation_pulse_width': pulse_width_seq, 
                'stimulation_frequency': frequency_seq, 
                # 'stimulation': stimulation_sequence,  #torch.Size([5, 2, 1000])
                'total_charge': total_charges,
                'phosphenes': phosphenes,#torch.Size([2, 1, 5, 256, 256])
                'reconstruction': reconstruction * cfg['circular_mask'], ##torch.Size([2, 1, 5, 128, 128])
                'input': frames * cfg['circular_mask'], #torch.Size([2, 1, 5, 128, 128])
                'input_resized': resize(frames *cfg['circular_mask'],
                                         (cfg['sequence_length'],*cfg['SPVsize']),interpolation='trilinear')} #torch.Size([2, 1, 5, 256, 256])

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}

        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

#added
def get_pipeline_unconstrained_video_reconstruction2(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # print('batch', len(batch), batch[0].shape, batch[1].shape)  #batch 2 torch.Size([4, 5, 128, 128]) torch.Size([4, 5, 128, 128])

        if isinstance(batch, list):
            image = batch[0]
            # print('islist')
        else:
            image = batch

        # print('image', image.shape) #torch.Size([4, 5, 128, 128])

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        # image, label = batch
        # print('image , label', image.shape, label.shape)
        # label = dilation3x3(label)
        # print('dilated label',  label.shape)
        

        # Forward pass
        simulator.reset()
        stimulation = encoder(image).permute(1, 0, 2)  # permute: (Batch,Channels=Time,Num_phos) -> (Channels=Time,Batch,Num_phos) #torch.Size([5, 4, 1000])
        # print('stim', stimulation.shape)
        # phosphenes = simulator(stimulation).unsqueeze(1)
        phosphenes = []
        for stim in stimulation:
            phosphenes.append(simulator(stim))  # simulator expects (Batch, Num_phosphenes)
        phosphenes = torch.stack(phosphenes, dim=1)# .unsqueeze(dim=1)  # Shape: (Batch, Channels=Time, Height, Width)
        # for i in range(phosphenes.shape[2]):
        #     plt.imsave(f'/home/burkuc/data/static/phos_v3_{i}.png', phosphenes[0,0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # plt.imsave(f'/home/burkuc/data/static/phos_3_all.png', phosphenes[0,0,:,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        #added
        # print('shape',phosphenes.shape[1])
        for i in range(phosphenes.shape[1]):
            plt.imsave(f'/home/burkuc/data/static/phos_v_{i}.png', phosphenes[0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # plt.imsave(f'/home/burkuc/data/static/phos_v_all.png', phosphenes[0,:,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # print('phosphenes', phosphenes.shape) #torch.Size([4, 5, 256, 256])
        reconstruction = decoder(phosphenes) * cfg['circular_mask']
        # print('recon', reconstruction.shape) #torch.Size([4, 5, 128, 128])
        
        

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction * cfg['circular_mask'],
            #    'target': label * cfg['circular_mask'],
            #    'target_resized': resize(label * cfg['circular_mask'], cfg['SPVsize'],),}
               'input': image * cfg['circular_mask'], #torch.Size([2, 1, 5, 128, 128])
               'input_resized': resize(image * cfg['circular_mask'], cfg['SPVsize'],),} #torch.Size([2, 1, 5, 256, 256])
                # 'input_resized': resize(image *cfg['circular_mask'],
                #                          (cfg['sequence_length'],*cfg['SPVsize']),interpolation='trilinear')} #torch.Size([2, 1, 5, 256, 256])


        # # Sample phosphenes and target at the centers of the phosphenes
        # out.update({'phosphene_centers': simulator.sample_centers(phosphenes) ,
        #             'target_centers': simulator.sample_centers(out['target_resized']) })

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight = 1/2) #1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight = 1/2) #cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

#added
def get_pipeline_unconstrained_video_reconstruction3(cfg):
    # def forward(batch, models, cfg, tb_writer,to_cpu=False):
    def forward(batch, models, cfg, to_cpu=False):
        # Unpack
        # print('batch', len(batch), batch[0].shape, batch[1].shape) 

        #added/changed
        if isinstance(batch, list):
            frames = batch[0]
            print('islist')
        else:
            frames = batch

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # bouncing , mode: recon >> batch dtype torch.float32 batch len 2 but its a tensor of torch.Size([2, 1, 5, 128, 128])  not list, so elements:  torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        # bouncing , mode: recon_pred or ade20k image,label >> batch 2 but it's a list of 2 elements(image, label or current,future), so elements:  torch.Size([2, 5, 128, 128]) torch.Size([2, 5, 128, 128])
        
        # print('frames', frames.shape, len(frames), frames[0].shape, frames[1].shape) # torch.Size([2, 1, 5, 128, 128]) 2 torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        
        # pdb.set_trace()
        # pdb.enable()
        # tb_writer.add_graph(encoder,frames)
        # Forward
        # simulator.reset() #RECONSIDER THIS -PAY ATTENTION-UNCOMMENTED,
        stimulation_sequence = encoder(frames) #.permute(1, 0, 2)  #  (Batch,Num_phos) 
        print('stimulation_sequence', stimulation_sequence.min(), stimulation_sequence.max())
        print('stimulation', stimulation_sequence)
        # print('stimulation_sequence reshaped', stimulation_sequence.shape) #torch.Size([5, 2, 1000])
        # phosphenes = []
        # for stim in stimulation_sequence:
        #     phosphenes.append(simulator(stim))  # simulator expects (Batch, Num_phosphenes)
        # phosphenes = torch.stack(phosphenes, dim=1).unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        # tb_writer.add_graph(simulator,stimulation_sequence)
        phosphenes = simulator(stimulation_sequence).unsqueeze(dim=1).unsqueeze(dim=1)   # Shape: (Batch, Channels=1, 1, Height, Width)??? do we want this or do we want another decoder thanzaho??
        print('phosphenes', phosphenes.min(), phosphenes.max())
        print('unqiue len',len(phosphenes.unique()))
        print('unqiue',phosphenes.unique())
        print('phosphenes', phosphenes)
        # pdb.set_trace()
        # pdb.enable()
        for i in range(phosphenes.shape[2]):
            plt.imsave(f'/home/burkuc/data/static/phos_v3n_{i}.png', phosphenes[0,0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # plt.imsave(f'/home/burkuc/data/static/phos_v_all.png', phosphenes[0,0,:,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # tb_writer.add_graph(decoder,phosphenes)
        reconstruction = decoder(phosphenes)  #torch.Size([2, 1, 1, 128, 128])
        # print('reconstruction ', reconstruction.shape) #torch.Size([2, 1, 1, 128, 128])
        # pdb.set_trace()
        # pdb.disable()

        out =  {'stimulation': stimulation_sequence, #torch.Size([2, 1000])
                'phosphenes': phosphenes,#torch.Size([2, 1, 1, 256, 256])
                'reconstruction': reconstruction * cfg['circular_mask'], ##torch.Size([2, 1, 1, 128, 128])
                # 'input': frames * cfg['circular_mask'],
                #added/changed
                'input': frames[:,:,-1,:,:].unsqueeze(dim=2) * cfg['circular_mask'], #torch.Size([2, 1, 1, 128, 128])
                # 'input_resized': resize(frames *cfg['circular_mask'],
                #                          (cfg['sequence_length'],*cfg['SPVsize']),interpolation='trilinear')} #torch.Size([2, 1, 5, 256, 256])
                #added/changed
                'input_resized': resize(frames[:,:,-1,:,:].unsqueeze(dim=2) *cfg['circular_mask'],
                                         (1,*cfg['SPVsize']),interpolation='trilinear')} #torch.Size([2, 1, 5, 256, 256])

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}

        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 / 2)

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=1 / 2)

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

def get_pipeline_unconstrained_video_reconstruction_rnn(cfg):
    def forward(batch, models, cfg, to_cpu=False):
    # def forward(batch, hidden_state, models, cfg, to_cpu=False):
        # Unpack
        # print('batch', batch.shape) 
        # frames = batch #torch.Size([2, 1, 10, 128, 128]) 
        #added/changed
        # if isinstance(batch, list):
        #     frames = batch[0]
        #     # print('islist')
        # else:
        #     frames = batch
        
        frames= batch.squeeze(1)
        # print('frames',frames.shape)

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # bouncing , mode: recon >> batch dtype torch.float32 batch len 2 but its a tensor of torch.Size([2, 1, 5, 128, 128])  not list, so elements:  torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        # bouncing , mode: recon_pred or ade20k image,label >> batch 2 but it's a list of 2 elements(image, label or current,future), so elements:  torch.Size([2, 5, 128, 128]) torch.Size([2, 5, 128, 128])
        
        # print('frames', frames.shape, len(frames), frames[0].shape, frames[1].shape) # torch.Size([2, 1, 5, 128, 128]) 2 torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        
        # pdb.set_trace()
        # pdb.enable()

        # Forward
        simulator.reset()
        hidden_state = (torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device']),torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device'])) 
        # print('enc out',encoder(frames, hidden_state)) #torch.Size([2, 1000])
        # pdb.set_trace()
        # pdb.enable()
        stimulation_sequence, rnn_output, hidden_state = encoder(frames, hidden_state)

        # print('stim seq', stimulation_sequence.shape) #, stimulation_sequence.min(), stimulation_sequence.max() )
        # print('rnn_output', rnn_output.shape, 'min', rnn_output.min(), 'max', rnn_output.max())
        # print('rrn last', rnn_output[-1:,:,:])
        # print('hidden0', hidden_state[0])
        # print('hidden0', hidden_state[0].shape, hidden_state[0].min(),hidden_state[0].max())
        # print('hidden1', hidden_state[1].shape, hidden_state[1].min(),hidden_state[1].max())
        # stimulation_sequence = encoder(frames).permute(1, 0, 2)  # permute: (Batch,Time,Num_phos) -> (Time,Batch,Num_phos) #

        # print('stimulation_sequence reshaped', stimulation_sequence.shape) #torch.Size([5, 2, 1000])
        # print('stimulation_sequence', stimulation_sequence.min(), stimulation_sequence.max())
        phosphenes = []
        # for stim in stimulation_sequence:
        #     phosphenes.append(simulator(stim))  # simulator expects (Batch, Num_phosphenes)
        # phosphenes = torch.stack(phosphenes, dim=1) #.unsqueeze(dim=1)  # Shape with unsqueeze: (Batch, Channels=1, Time, Height, Width)
        # print('phosphenes', phosphenes.min(), phosphenes.max())
        # print('phosphenes',phosphenes.shape)
        # for i in range(phosphenes.shape[2]):
        total_charges = []
        for stim in stimulation_sequence:
            phos, total_charge= simulator(stim)
            # phosphenes.append(simulator(amplitude, pulse_width, frequency))
            phosphenes.append(phos)
            # print('phos', phos.shape)
            total_charges.append(total_charge)
            # print('tc sh',total_charge.shape)


        phosphenes = torch.stack(phosphenes, dim=1)#.unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        #added
        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) #.unsqueeze(dim=1)
        #     plt.imsave(f'/home/burkuc/data/static/phos_v_{i}.png', phosphenes[0,0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # plt.imsave(f'/home/burkuc/data/static/phos_v_all.png', phosphenes[0,0,:,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # print('phosphenes reshaped', phosphenes.shape) #torch.Size([2, 1, 5, 256, 256])
        reconstruction = decoder(phosphenes)
        # print('reconstruction ', reconstruction.shape) #torch.Size([2, 1, 5, 128, 128])
        # print('circularm',cfg['circular_mask'].shape)
        # print('svp', cfg['SPVsize'])
        # print('frames',(frames * cfg['circular_mask']).shape)
        
        # print('resized', resize(frames *cfg['circular_mask'],
        #                                  (cfg['SPVsize'])).shape)
        # pdb.set_trace()
        # pdb.disable()

        out =  {'stimulation': stimulation_sequence, #torch.Size([5, 2, 1000])
                'total_charge': total_charges,
                'phosphenes': phosphenes,#torch.Size([2, 1, 5, 256, 256])
                'reconstruction': reconstruction * cfg['circular_mask'], ##torch.Size([2, 1, 5, 128, 128])
                'input': frames * cfg['circular_mask'], #torch.Size([2, 1, 5, 128, 128]) #([2, 10, 128, 128])
                'input_resized': resize(frames *cfg['circular_mask'],
                                         (cfg['SPVsize']))} #,interpolation='trilinear')} #torch.Size([2, 1, 5, 256, 256]) ##([2, 10, 256, 256])
                # 'input_resized': resize(frames *cfg['circular_mask'],
                #                          (cfg['sequence_length'],*cfg['SPVsize']),interpolation='trilinear')} #torch.Size([2, 1, 5, 256, 256])

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}

        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

#added
def get_pipeline_unconstrained_video_reconstruction_rnn_out3(cfg):
    def forward(batch, models, cfg, to_cpu=False):
    # def forward(batch, hidden_state, models, cfg, to_cpu=False):
        # Unpack
        # print('batch', batch.shape) 
        # frames = batch #torch.Size([2, 1, 10, 128, 128]) 
        #added/changed
        # if isinstance(batch, list):
        #     frames = batch[0]
        #     # print('islist')
        # else:
        #     frames = batch
        
        frames= batch.squeeze(1)
        # print('frames',frames.shape)

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # bouncing , mode: recon >> batch dtype torch.float32 batch len 2 but its a tensor of torch.Size([2, 1, 5, 128, 128])  not list, so elements:  torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        # bouncing , mode: recon_pred or ade20k image,label >> batch 2 but it's a list of 2 elements(image, label or current,future), so elements:  torch.Size([2, 5, 128, 128]) torch.Size([2, 5, 128, 128])
        
        # print('frames', frames.shape, len(frames), frames[0].shape, frames[1].shape) # torch.Size([2, 1, 5, 128, 128]) 2 torch.Size([1, 5, 128, 128]) torch.Size([1, 5, 128, 128])
        
        # pdb.set_trace()
        # pdb.enable()

        # Forward
        simulator.reset()
        hidden_state = (torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device']),torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device'])) 
        # print('enc out',encoder(frames, hidden_state)) #torch.Size([2, 1000])
        # pdb.set_trace()
        # pdb.enable()
        # stimulation_sequence, rnn_output, hidden_state = encoder(frames, hidden_state)
        amplitude_seq, pulse_width_seq, frequency_seq, rnn_output, hidden_state = encoder(frames, hidden_state)

        # amplitude_seq = amplitude_seq.permute(1, 0, 2)
        # pulse_width_seq = pulse_width_seq.permute(1, 0, 2)
        # frequency_seq = frequency_seq.permute(1, 0, 2)

        phosphenes = []
        total_charges = []
        for amplitude, pulse_width, frequency in zip(amplitude_seq, pulse_width_seq, frequency_seq):
            phos, total_charge= simulator(amplitude, pulse_width, frequency)
            # phosphenes.append(simulator(amplitude, pulse_width, frequency))
            phosphenes.append(phos)
            total_charges.append(total_charge)
            # print('phos', phos.shape)
            # print('tc sh',total_charge.shape)


        phosphenes = torch.stack(phosphenes, dim=1) #.unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        # print('phosphenes',phosphenes.shape)
        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) #.unsqueeze(dim=1)
        # print('total_charges',total_charges.shape)

        reconstruction = decoder(phosphenes)

        out =  {'stimulation_amplitude': amplitude_seq,
                'stimulation_pulse_width': pulse_width_seq, 
                'stimulation_frequency': frequency_seq, 
                # 'stimulation': stimulation_sequence,  #torch.Size([5, 2, 1000])
                'total_charge': total_charges,
                'phosphenes': phosphenes,#torch.Size([2, 1, 5, 256, 256])
                'reconstruction': reconstruction * cfg['circular_mask'], ##torch.Size([2, 1, 5, 128, 128])
                'input': frames * cfg['circular_mask'], #torch.Size([2, 1, 5, 128, 128]) #([2, 10, 128, 128])
                'input_resized': resize(frames *cfg['circular_mask'],
                                         (cfg['SPVsize']))} #,interpolation='trilinear')} #torch.Size([2, 1, 5, 256, 256]) ##([2, 10, 256, 256])
                # 'input_resized': resize(frames *cfg['circular_mask'],
                #                          (cfg['sequence_length'],*cfg['SPVsize']),interpolation='trilinear')} #torch.Size([2, 1, 5, 256, 256])

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}

        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func