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


# class L1FeatureLoss(object):
#     def __init__(self):
#         self.feature_extractor = model.VGGFeatureExtractor(device=device)
#         self.loss_fn = torch.nn.functional.l1_loss

#     def __call__(self, y_pred, y_true, ):
#         true_features = self.feature_extractor(y_true)
#         pred_features = self.feature_extractor(y_pred)
#         err = [self.loss_fn(pred, true) for pred, true in zip(pred_features, true_features)]
#         return torch.mean(torch.stack(err))



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

    elif cfg['model_architecture'] == 'recurrent_net':
        encoder, decoder = model.get_e2e_recurrent_net(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])

    elif cfg['model_architecture'] == 'recurrent_net_cons1_2fx':
        encoder, decoder = model.get_e2e_recurrent_net_cons1_2fx(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])

    elif cfg['model_architecture'] == 'recurrent_net_out3':
        encoder, decoder = model.get_e2e_recurrent_net_out3(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])

    elif cfg['model_architecture'] == 'recurrent_net_out32':
        encoder, decoder = model.get_e2e_recurrent_net_out32(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])

    elif cfg['model_architecture'] == 'end-to-end-autoencoder2': #could remove?
        encoder, decoder = model.get_e2e_autoencoder2(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate']) 
    elif cfg['model_architecture'] == 'zhao-autoencoder':
        encoder, decoder = model.get_Zhao_autoencoder(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])

    elif cfg['model_architecture'] == 'zhao-autoencoder2':
        encoder, decoder = model.get_Zhao_autoencoder2(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])

    elif cfg['model_architecture'] == 'zhao-autoencoder_out3':
        encoder, decoder = model.get_Zhao_autoencoder_out3(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])

    elif cfg['model_architecture'] == 'zhao-autoencoder_out32':
        encoder, decoder = model.get_Zhao_autoencoder_out32(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])

    elif cfg['model_architecture'] == 'zhao-autoencoder_cons1_2fx':
        encoder, decoder = model.get_Zhao_autoencoder_cons1_2fx(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    else:
        raise NotImplementedError

    simulator = get_simulator(cfg)


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


def get_training_pipeline(cfg):
    if cfg['pipeline'] == 'supervised-boundary-reconstruction':
        forward, lossfunc = get_pipeline_supervised_boundary_reconstruction(cfg)
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction(cfg)
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_rnn':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction_rnn(cfg)
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_rnn_cons1_2fx':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction_rnn_cons1_2fx(cfg)
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_rnn_out3':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction_rnn_out3(cfg) #using this
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction2':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction2(cfg)
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction3':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction3(cfg)
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_out3':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction_out3(cfg) #using this
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction_cons1_2fx':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction_cons1_2fx(cfg)   
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
        if isinstance(batch, list):
            frames = batch[0]
        else:
            frames = batch

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']


        # Forward
        simulator.reset()
        stimulation_sequence = encoder(frames).permute(1, 0, 2)  # permute: (Batch,Time,Num_phos) -> (Time,Batch,Num_phos) #

        phosphenes = []

        total_charges = []
        for stim in stimulation_sequence:
            phos, total_charge= simulator(stim)
            # phosphenes.append(simulator(amplitude, pulse_width, frequency))
            phosphenes.append(phos)
            total_charges.append(total_charge)
            # print('tc sh',total_charge.shape)


        phosphenes = torch.stack(phosphenes, dim=1).unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)

        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) 

        reconstruction = decoder(phosphenes)


        out =  {'stimulation': stimulation_sequence, #torch.Size([5, 2, 1000])
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
                          weight= cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

#added
def get_pipeline_unconstrained_video_reconstruction_cons1_2fx(cfg):
    def forward(batch, models, cfg, to_cpu=False):

        if isinstance(batch, list):
            frames = batch[0]
            # print('islist')
        else:
            frames = batch

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']


        # Forward
        simulator.reset()
        stimulation_sequence = encoder(frames).permute(1, 0, 2)  # permute: (Batch,Time,Num_phos) -> (Time,Batch,Num_phos) #

        phosphenes = []

        total_charges = []
        for stim in stimulation_sequence:
            if cfg['constrained_param']=='amplitude':
                phos, total_charge= simulator(stim, None, None)
            elif cfg['constrained_param']=='pulse_width':
                phos, total_charge= simulator(None, stim, None)
            elif cfg['constrained_param']=='frequency':
                phos, total_charge= simulator(None, None, stim)

            phosphenes.append(phos)
            total_charges.append(total_charge)


        phosphenes = torch.stack(phosphenes, dim=1).unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) 

        reconstruction = decoder(phosphenes)


        out =  {f"stimulation_{cfg['constrained_param']}": stimulation_sequence, #torch.Size([5, 2, 1000])
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
                          weight= cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

#added
def get_pipeline_unconstrained_video_reconstruction_out3(cfg):
    def forward(batch, models, cfg, to_cpu=False):

        if isinstance(batch, list):
            frames = batch[0]
            # print('islist')
        else:
            frames = batch

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']


        # Forward
        simulator.reset()
        amplitude_seq, pulse_width_seq, frequency_seq = encoder(frames)

        amplitude_seq = amplitude_seq.permute(1, 0, 2)
        pulse_width_seq = pulse_width_seq.permute(1, 0, 2)
        frequency_seq = frequency_seq.permute(1, 0, 2)


        phosphenes = []

        total_charges = []
        for amplitude, pulse_width, frequency in zip(amplitude_seq, pulse_width_seq, frequency_seq):
            phos, total_charge= simulator(amplitude, pulse_width, frequency)
            phosphenes.append(phos)
            total_charges.append(total_charge)



        phosphenes = torch.stack(phosphenes, dim=1).unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        
        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) #.unsqueeze(dim=1)

        reconstruction = decoder(phosphenes)


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

        if isinstance(batch, list):
            image = batch[0]
        else:
            image = batch


        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        

        # Forward pass
        simulator.reset()
        stimulation = encoder(image).permute(1, 0, 2)  # permute: (Batch,Channels=Time,Num_phos) -> (Channels=Time,Batch,Num_phos) #torch.Size([5, 4, 1000])

        phosphenes = []
        for stim in stimulation:
            phosphenes.append(simulator(stim))  # simulator expects (Batch, Num_phosphenes)
        phosphenes = torch.stack(phosphenes, dim=1)  # Shape: (Batch, Channels=Time, Height, Width)

        for i in range(phosphenes.shape[1]):
            plt.imsave(f'/home/burkuc/data/static/phos_v_{i}.png', phosphenes[0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 

        reconstruction = decoder(phosphenes) * cfg['circular_mask']
        
        

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
    def forward(batch, models, cfg, to_cpu=False):
        # Unpack

        if isinstance(batch, list):
            frames = batch[0]
 
        else:
            frames = batch

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        stimulation_sequence = encoder(frames) #.permute(1, 0, 2)  #  (Batch,Num_phos) 

        phosphenes = simulator(stimulation_sequence).unsqueeze(dim=1).unsqueeze(dim=1)   # Shape: (Batch, Channels=1, 1, Height, Width)??? do we want this or do we want another decoder thanzaho??

        for i in range(phosphenes.shape[2]):
            plt.imsave(f'/home/burkuc/data/static/phos_v3n_{i}.png', phosphenes[0,0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
 
        reconstruction = decoder(phosphenes)  #torch.Size([2, 1, 1, 128, 128])


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
        
        frames= batch.squeeze(1)

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Forward
        simulator.reset()
        hidden_state = (torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device']),torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device'])) 

        stimulation_sequence, rnn_output, hidden_state = encoder(frames, hidden_state)

        phosphenes = []

        total_charges = []
        for stim in stimulation_sequence:
            phos, total_charge= simulator(stim)
            phosphenes.append(phos)
            total_charges.append(total_charge)


        phosphenes = torch.stack(phosphenes, dim=1) # Shape: (Batch, Channels=1, Time, Height, Width)
  
        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) 
   
        reconstruction = decoder(phosphenes)


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

        
        frames= batch.squeeze(1)

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']


        # Forward
        simulator.reset()
        hidden_state = (torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device']),torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device'])) 

        amplitude_seq, pulse_width_seq, frequency_seq, rnn_output, hidden_state = encoder(frames, hidden_state)


        phosphenes = []
        total_charges = []
        for amplitude, pulse_width, frequency in zip(amplitude_seq, pulse_width_seq, frequency_seq):
            phos, total_charge= simulator(amplitude, pulse_width, frequency)
            phosphenes.append(phos)
            total_charges.append(total_charge)



        phosphenes = torch.stack(phosphenes, dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) #.unsqueeze(dim=1)

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

#added
def get_pipeline_unconstrained_video_reconstruction_rnn_cons1_2fx(cfg):
    def forward(batch, models, cfg, to_cpu=False):

        
        frames= batch.squeeze(1)

        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Forward
        simulator.reset()
        hidden_state = (torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device']),torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device'])) 

        stimulation_sequence, rnn_output, hidden_state = encoder(frames, hidden_state)

        phosphenes = []
        total_charges = []
        for stim in stimulation_sequence:
            if cfg['constrained_param']=='amplitude':
                phos, total_charge= simulator(stim, None, None)
            elif cfg['constrained_param']=='pulse_width':
                phos, total_charge= simulator(None, stim, None)
            elif cfg['constrained_param']=='frequency':
                phos, total_charge= simulator(None, None, stim)

            phosphenes.append(phos)
            total_charges.append(total_charge)

        phosphenes = torch.stack(phosphenes, dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        total_charges = torch.stack(total_charges, dim=1).permute(1,0,2) #.unsqueeze(dim=1)

        reconstruction = decoder(phosphenes)

        out =  {f"stimulation_{cfg['constrained_param']}": stimulation_sequence,
                # 'stimulation_amplitude': amplitude_seq,
                # 'stimulation_pulse_width': pulse_width_seq, 
                # 'stimulation_frequency': frequency_seq, 
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