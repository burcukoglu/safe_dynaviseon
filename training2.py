import os
import torch
import numpy as np
import pandas as pd
import pickle
from utils import resize, normalize, load_config, save_pickle, CustomSummaryTracker
import init_training2 as init_training
import argparse

#added
import pdb
import itertools
import matplotlib.pyplot as plt
import time

import torchvision
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(dataset, models, training_pipeline, logging, cfg, start):
    # Unpack
    trainloader = dataset['trainloader']
    example_batch = dataset['example_batch']
    optimizer = models['optimizer']
    compound_loss_func = training_pipeline['compound_loss_func']
    forward = training_pipeline['forward']
    training_loss = logging['training_loss']
    training_summary = logging['training_summary']
    validation_summary = logging['validation_summary']
    example_output = logging['example_output']
    tb_writer = logging['tensorboard_writer']

    
    print(f'parameters of encoder: {count_parameters(models["encoder"])}, decoder: {count_parameters(models["decoder"])}') #, simulator: {count_parameters(models["simulator"])}')
    if cfg['pipeline'] == "unconstrained-video-reconstruction_rnn":
        print(f'parameters of enc: {count_parameters(models["encoder"].enc)}, rnn: {count_parameters(models["encoder"].rnn)}') 


    # Make dir
    if not os.path.exists(cfg['save_path']):
        os.makedirs(cfg['save_path'])

    # Set torch deterministic (not possible on every GPU)
    if cfg['use_deterministic_algorithms']:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


    seed = cfg['seed']
    print('seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    
    

    # Training loop
    if start:
        best_validation_performance = np.inf
        epoch = 0
        print('STARTING TRAINING')
    else:
        print('RESUMING TRAINING')
        # epoch, batch_idx, best_validation_performance = load_models_to_resume_training(models, optimizer, cfg, prefix='last')
        epoch, best_validation_performance = load_models_to_resume_training(models, optimizer, cfg, prefix='last')
        print(f'loaded epoch, best_validation_performance: {epoch}, {best_validation_performance}')

    tb_writer.add_text('params_summary',str(cfg), epoch)

    # best_validation_performance = np.inf
    not_improved_count = 0
    # epoch = 0


    while epoch <= cfg['epochs'] and not_improved_count < cfg['early_stop_criterium']:
        print(f'\nepoch {epoch}')

        training_loss.reset()

        # print('trainloader', len(trainloader), len(trainloader[0]), trainloader)
        for batch_idx, batch in enumerate(trainloader, 1):  # range(100):
            # print('batch_idx', batch_idx)
            # first=time.time()
            # print("batchs",batch.shape)
            # for i in range(batch.shape[2]):
            #     plt.imsave(f'/home/burkuc/data/static/0kitti4{i}.png', batch[0,0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
            # pdb.set_trace()
            # pdb.enable()

            # print('exectime')
            # with profile(activities=[
            #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof: #profiler mesures compute by default
            #     with record_function("model_inference"):
            #         # model(inputs)
            #         model_output = forward(batch, models, cfg)

            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            # print('CPU')
            # with profile(activities=[ProfilerActivity.CPU], #profile_memory so its memory profiling not compute
            #     profile_memory=True, record_shapes=True) as prof:
            #     # model(inputs)
            #     model_output = forward(batch, models, cfg)

            # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
            
            # print('CUDA')
            # with profile(activities=[ProfilerActivity.CUDA],
            #     profile_memory=True, record_shapes=True) as prof:
            #     # model(inputs)
            #     model_output = forward(batch, models, cfg)

            # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

            # print('cpu CUDA')
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #     profile_memory=True, record_shapes=True) as prof:
            #     # model(inputs)
            #     model_output = forward(batch, models, cfg)

            # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            
            
            
            # print('json')
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            #     # model(inputs)
            #     model_output = forward(batch, models, cfg)

            # prof.export_chrome_trace("trace2.json")

            # exit()
            

            # Forward pass
            model_output = forward(batch, models, cfg)
            total_loss = compound_loss_func(model_output)['total']

            # print(models['simulator'].activation.state.shape)
            # pdb.set_trace()
            # pdb.enable()

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward(retain_graph=False)
            optimizer.step()

            # Track the loss summary
            training_loss.update(compound_loss_func.items())

            # end=time.time()
            # print('dur batch forward', end-first)
            if batch_idx % (len(trainloader) // cfg['trainstats_per_epoch']) == 0:
                # Get the average loss over last batches
                training_performance = training_loss.get()
                training_loss.reset()

                # Store and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                training_summary.update({**timestamp, **training_performance})
                tb_writer.add_scalars('loss/training', training_performance, timestamp['samples'])
                tb_writer.add_scalar('stimulation/min_amp', model_output["stimulation"].min(), timestamp['samples'])
                tb_writer.add_scalar('stimulation/max_amp', model_output["stimulation"].max(), timestamp['samples'])
                print(timestamp['timestamp'] + '-tr ' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in training_performance.items()]))

                # Process example batch
                with torch.no_grad():
                    model_output = forward(example_batch, models, cfg, to_cpu=True)

                # Store examples in the summary trackers
                example_output.update(model_output)
                for key in cfg['save_output']:
                    shape = model_output[key].shape
                    # print(key, shape)
                    # print('timestamp', timestamp['samples'])
                    if len(shape) == 4:  # Image batch (N,C,H,W)
                        # tb_writer.add_images(key,
                        #                      normalize(model_output[key]),  # (scale to range [0, 1])
                        #                      timestamp['samples'], dataformats='NCHW')
                        #added/changed
                        if shape[1]==cfg['sequence_length']:  # Image batch (N,C,H,W)
                            img_batch = model_output[key][0].unsqueeze(dim=0).permute(1,0,2,3) 
                            tb_writer.add_images(key,
                                             normalize(img_batch),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                        elif shape[1]==1: 
                            tb_writer.add_images(key,
                                                normalize(model_output[key]),  # (scale to range [0, 1])
                                                timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 5: # Video batch (N, C, T, H, W)
                        img_batch = model_output[key][0].permute(1,0,2,3) # First video as img batch
                        tb_writer.add_images(key,
                                             normalize(img_batch),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 2: # (N, P)
                        tb_writer.add_histogram(key, model_output[key]) # Stimulation
                    #added
                    elif len(shape) == 3: # (N, P) torch.Size([10, 2, 1000])
                        tb_writer.add_histogram(key+"_all", model_output[key][:,0,:], timestamp['samples']) # Stimulation
                        for i in range(shape[0]):
                        #     tb_writer.add_histogram(key+str(i), model_output[key][i,0,:], timestamp['samples']) # Stimulation
                            tb_writer.add_histogram(key+'_step', model_output[key][i,0,:], timestamp['samples']+i) 

            if batch_idx % (len(trainloader) // cfg['validations_per_epoch']) == 0:
                # Run validation loop
                validation_performance = validation(dataset, models, training_pipeline, logging, cfg)

                # Track and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                validation_summary.update({**timestamp, **validation_performance})
                tb_writer.add_scalars('/loss/validation', validation_performance, timestamp['samples'])
                print(timestamp['timestamp'] + '-val' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in validation_performance.items()]))

                if validation_performance['total'] < best_validation_performance:
                    best_validation_performance = validation_performance['total']
                    print("Model has improved")
                    not_improved_count = 0
                    save_models(models, cfg, prefix='best')
                    #added
                    tb_writer.add_text('best_models',f'best models saved at epoch:{epoch}, batch_idx:{batch_idx}', best_validation_performance)

                else:
                    not_improved_count += 1
                    print(f"Not improved during last {not_improved_count} validations")
                    #added
                    # save_models(models, cfg, prefix='last')
                    # tb_writer.add_text('last_models',f'last models saved at epoch:{epoch}, batch_idx:{batch_idx}', validation_performance['total'])

                    if not_improved_count >= cfg['early_stop_criterium']:
                        break
        
        #added
        save_models_to_resume_training(models, optimizer, epoch, batch_idx, best_validation_performance, cfg, prefix='last')

        epoch += 1

    print("--- Finished training ---\n")

#added
def train_rnn(dataset, models, training_pipeline, logging, cfg, start):
    # Unpack
    trainloader = dataset['trainloader']
    example_batch = dataset['example_batch']
    optimizer = models['optimizer']
    compound_loss_func = training_pipeline['compound_loss_func']
    forward = training_pipeline['forward']
    training_loss = logging['training_loss']
    training_summary = logging['training_summary']
    validation_summary = logging['validation_summary']
    example_output = logging['example_output']
    tb_writer = logging['tensorboard_writer']

    print(f'parameters of encoder: {count_parameters(models["encoder"])}, decoder: {count_parameters(models["decoder"])}')#, simulator: {count_parameters(models["simulator"])}')
    print(f'parameters of enc: {count_parameters(models["encoder"].enc)}, rnn: {count_parameters(models["encoder"].rnn)}') 
    # exit() 
    # Make dir
    if not os.path.exists(cfg['save_path']):
        os.makedirs(cfg['save_path'])

    # Set torch deterministic (not possible on every GPU)
    if cfg['use_deterministic_algorithms']:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


    seed = cfg['seed']
    print('seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    

    # Training loop
    if start:
        best_validation_performance = np.inf
        epoch = 0
        print('STARTING TRAINING')
    else:
        print('RESUMING TRAINING')
        # epoch, batch_idx, best_validation_performance = load_models_to_resume_training(models, optimizer, cfg, prefix='last')
        epoch, best_validation_performance = load_models_to_resume_training(models, optimizer, cfg, prefix='last')
        print(f'loaded epoch, best_validation_performance: {epoch}, {best_validation_performance}')
    
    tb_writer.add_text('params_summary',str(cfg), epoch)

    not_improved_count = 0
    while epoch <= cfg['epochs'] and not_improved_count < cfg['early_stop_criterium']:
        print(f'\nepoch {epoch}') 

        training_loss.reset()
        # hidden_state = (torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device']),torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device'])) 
        # print('hiddenstate', hidden_state[0].shape, hidden_state[1].shape)

        # print('trainloader', len(trainloader), len(trainloader[0]), trainloader)
        for batch_idx, batch in enumerate(trainloader, 1):  # range(100):
            print('batch_idx', batch_idx)
            # print("batchs",batch.shape)
            # for i in range(batch.shape[2]):
            #     plt.imsave(f'/home/burkuc/data/static/0kitti4{i}.png', batch[0,0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
            # pdb.set_trace()
            # pdb.enable()
            # Forward pass
            # model_output = forward(batch, hidden_state, models, cfg)
            model_output = forward(batch, models, cfg)
            total_loss = compound_loss_func(model_output)['total']

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward(retain_graph=False)
            optimizer.step()

            # Track the loss summary
            training_loss.update(compound_loss_func.items())

            if batch_idx % (len(trainloader) // cfg['trainstats_per_epoch']) == 0:
                # Get the average loss over last batches
                training_performance = training_loss.get()
                training_loss.reset()

                # Store and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                print('timestamp', timestamp, 'samples', timestamp['samples'])
                training_summary.update({**timestamp, **training_performance})
                tb_writer.add_scalars('loss/training', training_performance, timestamp['samples'])
                tb_writer.add_scalar('stimulation/min_amp', model_output["stimulation"].min(), timestamp['samples'])
                tb_writer.add_scalar('stimulation/max_amp', model_output["stimulation"].max(), timestamp['samples'])
                print(timestamp['timestamp'] + '-tr ' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in training_performance.items()]))

                # Process example batch
                with torch.no_grad():
                    model_output = forward(example_batch, models, cfg, to_cpu=True)
                    # model_output = forward(example_batch, hidden_state, models, cfg, to_cpu=True)

                # Store examples in the summary trackers
                example_output.update(model_output)
                for key in cfg['save_output']:
                    shape = model_output[key].shape
                    # print(key, shape)
                    # print('timestamp', timestamp['samples'])
                    if len(shape) == 4:  # Image batch (N,C,H,W)
                        # tb_writer.add_images(key,
                        #                      normalize(model_output[key]),  # (scale to range [0, 1])
                        #                      timestamp['samples'], dataformats='NCHW')
                        #added/changed
                        if shape[1]==cfg['sequence_length']:  # Image batch (N,C,H,W)
                            img_batch = model_output[key][0].unsqueeze(dim=0).permute(1,0,2,3) 
                            tb_writer.add_images(key,
                                             normalize(img_batch),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                        elif shape[1]==1: 
                            tb_writer.add_images(key,
                                                normalize(model_output[key]),  # (scale to range [0, 1])
                                                timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 5: # Video batch (N, C, T, H, W)
                        img_batch = model_output[key][0].permute(1,0,2,3) # First video as img batch
                        tb_writer.add_images(key,
                                             normalize(img_batch),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 2: # (N, P)
                        tb_writer.add_histogram(key, model_output[key]) # Stimulation
                    #added
                    elif len(shape) == 3: # (N, P) torch.Size([10, 2, 1000])
                        tb_writer.add_histogram(key+"_all", model_output[key][:,0,:], timestamp['samples']) # Stimulation
                        for i in range(shape[0]):
                        #     tb_writer.add_histogram(key+str(i), model_output[key][i,0,:], timestamp['samples']) # Stimulation
                            tb_writer.add_histogram(key+'_step', model_output[key][i,0,:], timestamp['samples']+i) 

            if batch_idx % (len(trainloader) // cfg['validations_per_epoch']) == 0:
                # Run validation loop
                validation_performance = validation(dataset, models, training_pipeline, logging, cfg)

                # Track and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                validation_summary.update({**timestamp, **validation_performance})
                tb_writer.add_scalars('/loss/validation', validation_performance, timestamp['samples'])
                print(timestamp['timestamp'] + '-val' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in validation_performance.items()]))

                if validation_performance['total'] < best_validation_performance:
                    best_validation_performance = validation_performance['total']
                    print("Model has improved")
                    not_improved_count = 0
                    save_models(models, cfg, prefix='best')
                    #added
                    tb_writer.add_text('best_models',f'best models saved at epoch:{epoch}, batch_idx:{batch_idx}', best_validation_performance)

                else:
                    not_improved_count += 1
                    print(f"Not improved during last {not_improved_count} validations")
                    #added
                    # tb_writer.add_text('last_models',f'last models saved at epoch:{epoch}, batch_idx:{batch_idx}', validation_performance['total'])
                    
                    if not_improved_count >= cfg['early_stop_criterium']:
                        break
                
        save_models_to_resume_training(models, optimizer, epoch, batch_idx, best_validation_performance, cfg, prefix='last')
        
        #added
        
        epoch += 1

    print("--- Finished training ---\n")

#added
def train2(dataset, models, training_pipeline, logging, cfg):
    # Unpack
    trainloader = dataset['trainloader']
    example_batch = dataset['example_batch']
    optimizer = models['optimizer']
    compound_loss_func = training_pipeline['compound_loss_func']
    forward = training_pipeline['forward']
    training_loss = logging['training_loss']
    training_summary = logging['training_summary']
    validation_summary = logging['validation_summary']
    example_output = logging['example_output']
    tb_writer = logging['tensorboard_writer']

    # Make dir
    if not os.path.exists(cfg['save_path']):
        os.makedirs(cfg['save_path'])

    # Set torch deterministic (not possible on every GPU)
    if cfg['use_deterministic_algorithms']:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
    
    seed = cfg['seed']
    print('seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    # Training loop
    best_validation_performance = np.inf
    not_improved_count = 0
    epoch = 0

            
    while epoch <= cfg['epochs'] and not_improved_count < cfg['early_stop_criterium']:
        print(f'\nepoch {epoch}')

        training_loss.reset()

        # print('trainloader', len(trainloader), len(trainloader[0]), trainloader)
        # for batch_idx, batch in enumerate(trainloader, 1): 
        #ADDED/CHANGED
        for batch_idx, batch_orig in enumerate(trainloader, 1):  # range(100):

            #ADDED TO take resetting -REVISIT When dataset becomes dynamic!
            # models['simulator'].reset()
            
            #ADDED/CHANGED IF-ELSE for sequence length so same batch is taken multipel times (before it was only the section under else)
            if cfg['sequence_length']>1 and cfg['dataset']=='ADE50K' and 'zhao' in cfg['model_architecture']:
                # black = torch.zeros(batch.shape[0],batch.shape[1], 1, batch.shape[3],batch.shape[4])
                black = torch.zeros(batch_orig.shape).to(cfg['device'])
                # print('black', black.shape)
                
                
                all_indices = list(range(cfg['sequence_length']))
                # print('all_indices', all_indices)

                #added
                models['simulator'].reset()
                #ADDED
                optimizer.zero_grad()
                print('resetted simulator & zero grad')

                for slice_idx in range(1,cfg['sequence_length'] * 2):
                    batch = torch.clone(batch_orig)

                    if slice_idx<cfg['sequence_length']:
                        black_indices = all_indices[:-slice_idx]
                        # print('i', slice_idx, 'black_indices',black_indices)
                        # print('old batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
                        batch[:,:,black_indices,:,:] = black[:,:,black_indices,:,:]
                        # print('new batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
                    elif slice_idx>cfg['sequence_length']:
                        black_indices = all_indices[cfg['sequence_length'] * 2 - slice_idx:]
                        # print('i', slice_idx, 'black_indices',black_indices)
                        # print('old batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
                        batch[:,:,black_indices,:,:] = black[:,:,black_indices,:,:]
                        # print('new batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
                    
                    # for k in range(batch.shape[2]):
                    #     plt.imsave(f'/home/burkuc/data/static/batch_{all_indices[-1]+1}_{slice_idx}_{k}.png', batch[0,0,k,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
                    
                    #UNCOMMENT
                    # # Forward pass
                    # model_output = forward(batch, models, cfg)
                    # total_loss = compound_loss_func(model_output)['total']

                    # # Backward pass
                    # optimizer.zero_grad()
                    # # total_loss.backward(retain_graph=False)
                    # total_loss.backward(retain_graph=False) #REVISIT
                    # optimizer.step()

                    # # Track the loss summary
                    # training_loss.update(compound_loss_func.items())

                    model_output = forward(batch, models, cfg) #, tb_writer)
                    total_loss = compound_loss_func(model_output)['total']

                    # Backward pass
                    # optimizer.zero_grad() #REMOVED BE CAREFUL
                    if (slice_idx)%(cfg['sequence_length'] * 2 - 1)==0:
                        print(slice_idx,'retain false')
                        # optimizer.zero_grad() 
                        total_loss.backward(retain_graph=False) 
                        optimizer.step() #only in the last one.
                    else:
                        
                        print(slice_idx,'retain true')
                        total_loss.backward(retain_graph=True) #FALSE BE CAREFUL
                    # optimizer.step()

                    # Track the loss summary
                    training_loss.update(compound_loss_func.items()) # HERE OR ONLY AFTER STEP?
                
            else:

                #added
                models['simulator'].reset()
                batch = batch_orig

                # Forward pass
                model_output = forward(batch, models, cfg)
                total_loss = compound_loss_func(model_output)['total']

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward(retain_graph=False)
                optimizer.step()

                # Track the loss summary
                training_loss.update(compound_loss_func.items())

            if batch_idx % (len(trainloader) // cfg['trainstats_per_epoch']) == 0:
                # Get the average loss over last batches
                training_performance = training_loss.get()
                training_loss.reset()

                # Store and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                training_summary.update({**timestamp, **training_performance})
                tb_writer.add_scalars('loss/training', training_performance, timestamp['samples'])
                print(timestamp['timestamp'] + '-tr ' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in training_performance.items()]))

                # Process example batch
                with torch.no_grad():
                    model_output = forward(example_batch, models, cfg, to_cpu=True)

                # Store examples in the summary trackers
                example_output.update(model_output)
                for key in cfg['save_output']:
                    shape = model_output[key].shape
                    if len(shape) == 4:  # Image batch (N,C,H,W)
                        # tb_writer.add_images(key,
                        #                      normalize(model_output[key]),  # (scale to range [0, 1])
                        #                      timestamp['samples'], dataformats='NCHW')
                        #added/changed
                        if shape[1]==cfg['sequence_length']:  # Image batch (N,C,H,W)
                            img_batch = model_output[key][0].unsqueeze(dim=0).permute(1,0,2,3) 
                            tb_writer.add_images(key,
                                             normalize(img_batch),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                        elif shape[1]==1: 
                            tb_writer.add_images(key,
                                                normalize(model_output[key]),  # (scale to range [0, 1])
                                                timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 5: # Video batch (N, C, T, H, W)
                        img_batch = model_output[key][0].permute(1,0,2,3) # First video as img batch
                        tb_writer.add_images(key,
                                             normalize(img_batch),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 2: # (N, P)
                        tb_writer.add_histogram(key, model_output[key]) # Stimulation

            if batch_idx % (len(trainloader) // cfg['validations_per_epoch']) == 0:
                # Run validation loop
                # validation_performance = validation(dataset, models, training_pipeline, logging, cfg)
                #added/changed
                validation_performance = validation2(dataset, models, training_pipeline, logging, cfg)

                # Track and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                validation_summary.update({**timestamp, **validation_performance})
                tb_writer.add_scalars('/loss/validation', validation_performance, timestamp['samples'])
                print(timestamp['timestamp'] + '-val' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in validation_performance.items()]))

                if validation_performance['total'] < best_validation_performance:
                    best_validation_performance = validation_performance['total']
                    print("Model has improved")
                    not_improved_count = 0
                    save_models(models, cfg, prefix='best')

                else:
                    not_improved_count += 1
                    print(f"Not improved during last {not_improved_count} validations")
                    if not_improved_count >= cfg['early_stop_criterium']:
                        break
        epoch += 1

    print("--- Finished training ---\n")


def train3(dataset, models, training_pipeline, logging, cfg):
    # Unpack
    trainloader = dataset['trainloader']
    example_batch = dataset['example_batch']
    optimizer = models['optimizer']
    compound_loss_func = training_pipeline['compound_loss_func']
    forward = training_pipeline['forward']
    training_loss = logging['training_loss']
    training_summary = logging['training_summary']
    validation_summary = logging['validation_summary']
    example_output = logging['example_output']
    tb_writer = logging['tensorboard_writer']

    # Make dir
    if not os.path.exists(cfg['save_path']):
        os.makedirs(cfg['save_path'])

    # Set torch deterministic (not possible on every GPU)
    if cfg['use_deterministic_algorithms']:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)

    # Training loop
    best_validation_performance = np.inf
    not_improved_count = 0
    epoch = 0

    #added
    # if cfg['sequence_length']>1 and cfg['dataset']=='ADE50K':
    #     sequence_range = range(cfg['sequence_length'])
    #     print('sequence_range',sequence_range)
    #     combinations = []
    #     for r in range(1,cfg['sequence_length']): #combinations
    #         combinations.extend(list(itertools.combinations(sequence_range, r=r)))
    #         print(r, combinations)
    #     print('final', combinations)
            
    while epoch <= cfg['epochs'] and not_improved_count < cfg['early_stop_criterium']:
        print(f'\nepoch {epoch}')

        training_loss.reset()

        # print('trainloader', len(trainloader), len(trainloader[0]), trainloader)
        # for batch_idx, batch in enumerate(trainloader, 1): 
        #ADDED/CHANGED
        for batch_idx, batch_orig in enumerate(trainloader, 1):  # range(100):

            # #ADDED TO take resetting -REVISIT When dataset becomes dynamic!
            # # models['simulator'].reset()
            
            # #ADDED/CHANGED IF-ELSE for sequence length so same batch is taken multipel times (before it was only the section under else)
            # if cfg['sequence_length']>1 and cfg['dataset']=='ADE50K' and 'zhao' in cfg['model_architecture']:
            #     # black = torch.zeros(batch.shape[0],batch.shape[1], 1, batch.shape[3],batch.shape[4])
            #     black = torch.zeros(batch_orig.shape).to(cfg['device'])
            #     # print('black', black.shape)
                
                
            #     all_indices = list(range(cfg['sequence_length']))
            #     # print('all_indices', all_indices)

            #     #added
            #     models['simulator'].reset()
                
            #     for slice_idx in range(1,cfg['sequence_length'] * 2):
            #         batch = torch.clone(batch_orig)

            #         if slice_idx<cfg['sequence_length']:
            #             black_indices = all_indices[:-slice_idx]
            #             # print('i', slice_idx, 'black_indices',black_indices)
            #             # print('old batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
            #             batch[:,:,black_indices,:,:] = black[:,:,black_indices,:,:]
            #             # print('new batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
            #         elif slice_idx>cfg['sequence_length']:
            #             black_indices = all_indices[cfg['sequence_length'] * 2 - slice_idx:]
            #             # print('i', slice_idx, 'black_indices',black_indices)
            #             # print('old batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
            #             batch[:,:,black_indices,:,:] = black[:,:,black_indices,:,:]
            #             # print('new batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
                    
            #         # for k in range(batch.shape[2]):
            #         #     plt.imsave(f'/home/burkuc/data/static/batch_{all_indices[-1]+1}_{slice_idx}_{k}.png', batch[0,0,k,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
                    
                
            #         # Forward pass
            #         model_output = forward(batch, models, cfg)
            #         total_loss = compound_loss_func(model_output)['total']

            #         # Backward pass
            #         optimizer.zero_grad()
            #         # total_loss.backward(retain_graph=False)
            #         total_loss.backward(retain_graph=False) #REVISIT
            #         optimizer.step()

            #         # Track the loss summary
            #         training_loss.update(compound_loss_func.items())
               
            # else:

            #     #added
            #     models['simulator'].reset()
            #     batch = batch_orig

            #     # Forward pass
            #     model_output = forward(batch, models, cfg)
            #     total_loss = compound_loss_func(model_output)['total']

            #     # Backward pass
            #     optimizer.zero_grad()
            #     total_loss.backward(retain_graph=False)
            #     optimizer.step()

            #     # Track the loss summary
            #     training_loss.update(compound_loss_func.items())
            #added
            batch = batch_orig

            # Forward pass
            with torch.autograd.set_detect_anomaly(True):
                print('batch_idx', batch_idx)
                if (batch_idx-1)%(cfg['sequence_length'] * 2 - 1)==0:
                    print('resetting')
                    models['simulator'].reset()
                    optimizer.zero_grad()
                
                # batch = batch_orig
            
                model_output = forward(batch, models, cfg) #, tb_writer)
                total_loss = compound_loss_func(model_output)['total']

                # Backward pass
                # optimizer.zero_grad() #REMOVED BE CAREFUL
                if (batch_idx)%(cfg['sequence_length'] * 2 - 1)==0:
                    print('retain false')
                    # optimizer.zero_grad() 
                    total_loss.backward(retain_graph=False) 
                    optimizer.step() #only in the last one.
                else:
                    
                    print('retain true')
                    total_loss.backward(retain_graph=True) #FALSE BE CAREFUL
                # optimizer.step()

            # Track the loss summary
            training_loss.update(compound_loss_func.items())

            if batch_idx % (len(trainloader) // cfg['trainstats_per_epoch']) == 0:
                # Get the average loss over last batches
                training_performance = training_loss.get()
                training_loss.reset()

                # Store and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                training_summary.update({**timestamp, **training_performance})
                tb_writer.add_scalars('loss/training', training_performance, timestamp['samples'])
                print(timestamp['timestamp'] + '-tr ' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in training_performance.items()]))

                # Process example batch
                with torch.no_grad():
                    model_output = forward(example_batch, models, cfg, to_cpu=True)

                # Store examples in the summary trackers
                example_output.update(model_output)
                for key in cfg['save_output']:
                    shape = model_output[key].shape
                    if len(shape) == 4:  # Image batch (N,C,H,W)
                        # tb_writer.add_images(key,
                        #                      normalize(model_output[key]),  # (scale to range [0, 1])
                        #                      timestamp['samples'], dataformats='NCHW')
                        #added/changed
                        if shape[1]==cfg['sequence_length']:  # Image batch (N,C,H,W)
                            img_batch = model_output[key][0].unsqueeze(dim=0).permute(1,0,2,3) 
                            tb_writer.add_images(key,
                                             normalize(img_batch),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                        elif shape[1]==1: 
                            tb_writer.add_images(key,
                                                normalize(model_output[key]),  # (scale to range [0, 1])
                                                timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 5: # Video batch (N, C, T, H, W)
                        img_batch = model_output[key][0].permute(1,0,2,3) # First video as img batch
                        tb_writer.add_images(key,
                                             normalize(img_batch),  # (scale to range [0, 1])
                                             timestamp['samples'], dataformats='NCHW')
                    elif len(shape) == 2: # (N, P)
                        tb_writer.add_histogram(key, model_output[key]) # Stimulation

            if batch_idx % (len(trainloader) // cfg['validations_per_epoch']) == 0:
                # Run validation loop
                # validation_performance = validation(dataset, models, training_pipeline, logging, cfg)
                #added/changed
                validation_performance = validation2(dataset, models, training_pipeline, logging, cfg)

                # Track and print the training performance
                timestamp = get_timestamp(epoch, batch_idx, total_batches_per_epoch=len(trainloader),
                                          batch_size=cfg['batch_size'])
                validation_summary.update({**timestamp, **validation_performance})
                tb_writer.add_scalars('/loss/validation', validation_performance, timestamp['samples'])
                print(timestamp['timestamp'] + '-val' + ''.join(
                    ['  {:.8}:  {:.5f}'.format(k, v) for k, v in validation_performance.items()]))

                if validation_performance['total'] < best_validation_performance:
                    best_validation_performance = validation_performance['total']
                    print("Model has improved")
                    not_improved_count = 0
                    save_models(models, cfg, prefix='best')

                else:
                    not_improved_count += 1
                    print(f"Not improved during last {not_improved_count} validations")
                    if not_improved_count >= cfg['early_stop_criterium']:
                        break
        epoch += 1

    print("--- Finished training ---\n")

def validation(dataset, models, training_pipeline, logging, cfg):
    # Unpack
    valloader = dataset['valloader']
    compound_loss_func = training_pipeline['compound_loss_func']
    forward = training_pipeline['forward']
    validation_loss = logging['validation_loss']

    # Set models to eval
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.eval()

    # Loop over validation set and calculate validation loss
    validation_loss.reset()
    # hidden_state = (torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device']),torch.zeros(cfg["rnn_num_layers"], cfg["batch_size"], cfg["n_electrodes"]).to(cfg['device'])) 

    for batch_idx, batch in enumerate(valloader, 1):  # range(100):

        # Forward pass
        with torch.no_grad():
            model_output = forward(batch, models, cfg)
            # model_output = forward(batch, hidden_state, models, cfg)
            loss = compound_loss_func(model_output)

        # Update running stats
        validation_loss.update(compound_loss_func.items())

    # Get the average loss over last batches
    validation_performance = validation_loss.get()

    # Reset models to training mode
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.train()
    return validation_performance

#added
def validation2(dataset, models, training_pipeline, logging, cfg):
    # Unpack
    valloader = dataset['valloader']
    compound_loss_func = training_pipeline['compound_loss_func']
    forward = training_pipeline['forward']
    validation_loss = logging['validation_loss']

    # Set models to eval
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.eval()

    # Loop over validation set and calculate validation loss
    validation_loss.reset()
    # for batch_idx, batch in enumerate(valloader, 1):  # range(100):
    #CHANGED
    for batch_idx, batch_orig in enumerate(valloader, 1):  # range(100):
        #ADDED TO take resetting -REVISIT When dataset becomes dynamic!
        models['simulator'].reset()
        
        #ADDED/CHANGED IF-ELSE for sequence length so same batch is taken multipel times (before it was only the section under else)
        if cfg['sequence_length']>1 and cfg['dataset']=='ADE50K' and 'zhao' in cfg['model_architecture']:
            # black = torch.zeros(batch.shape[0],batch.shape[1], 1, batch.shape[3],batch.shape[4])
            black = torch.zeros(batch_orig.shape).to(cfg['device'])
            # print('black', black.shape)
            
            # for comb in combinations:
            #     pdb.set_trace()
            #     pdb.enable()
            #     print('old batch', batch[:,:,comb,:,:].min(), batch[:,:,comb,:,:].max())
            #     batch[:,:,comb,:,:]=0
            #     # batch[:,:,comb,:,:]=black[:,:,comb,:,:] #REVISIT
            #     print('new batch', batch.shape, batch[:,:,comb,:,:].min(), batch[:,:,comb,:,:].max() )
            
            all_indices = list(range(cfg['sequence_length']))
            # print('all_indices', all_indices)
            for slice_idx in range(1,cfg['sequence_length'] * 2):
                batch = torch.clone(batch_orig)

                if slice_idx<cfg['sequence_length']:
                    black_indices = all_indices[:-slice_idx]
                    # print('i', slice_idx, 'black_indices',black_indices)
                    # print('old batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
                    batch[:,:,black_indices,:,:] = black[:,:,black_indices,:,:]
                    # print('new batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
                elif slice_idx>cfg['sequence_length']:
                    black_indices = all_indices[cfg['sequence_length'] * 2 - slice_idx:]
                    # print('i', slice_idx, 'black_indices',black_indices)
                    # print('old batch', batch.shape, batch[:,:,black_indices,:,:].min(), batch[:,:,black_indices,:,:].max() )
                    batch[:,:,black_indices,:,:] = black[:,:,black_indices,:,:]
                
                with torch.no_grad():
                    model_output = forward(batch, models, cfg)
                    loss = compound_loss_func(model_output)

                # Update running stats
                validation_loss.update(compound_loss_func.items())

        else:
            #added
            batch = batch_orig
            # Forward pass
            with torch.no_grad():
                model_output = forward(batch, models, cfg)
                loss = compound_loss_func(model_output)

            # Update running stats
            validation_loss.update(compound_loss_func.items())

    # Get the average loss over last batches
    validation_performance = validation_loss.get()

    # Reset models to training mode
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.train()
    return validation_performance

def get_timestamp(epoch, batch_idx, total_batches_per_epoch, batch_size):
    timestamp = {'timestamp': f'E{epoch:02d}-B{batch_idx:03d}',
                 'epochs': epoch + batch_idx / total_batches_per_epoch,
                 'samples': batch_size * (batch_idx + total_batches_per_epoch * epoch),}
    return timestamp


def save_models(models, cfg, prefix='best'):
    # Create directory if not exists
    path = os.path.join(cfg['save_path'], 'checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)

    # Save model parameters
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            fn = os.path.join(path, f'{prefix}_{name}.pth')
            torch.save(model.state_dict(), fn)
            print(f"Saving parameters to {fn}")

#added
def save_models_to_resume_training(models, optimizer, epoch, batch_idx, best_validation_performance, cfg, prefix='last'):
    # Create directory if not exists
    path = os.path.join(cfg['save_path'], 'checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)

    
    state = {
    'epoch': epoch,
    'batch_idx': batch_idx,
    'state_dict_encoder': models["encoder"].state_dict(),
    'state_dict_decoder': models["decoder"].state_dict(),
    'optimizer': optimizer.state_dict(),
    'cfg': cfg,
    'best_validation_performance': best_validation_performance
    }

    # Save model parameters
    fn = os.path.join(path, f'{prefix}_checkpoint.pth')
    torch.save(state, fn)
    print(f"Saving parameters to {fn}")

#added
def load_models_to_resume_training(models, optimizer, cfg, prefix='last'):
    fn = os.path.join(cfg['save_path'], 'checkpoints', f'{prefix}_checkpoint.pth')
    state = torch.load(fn, map_location=cfg['device'])
    for name, model in models.items():
        # print('name, model', name, model)
        if isinstance(model, torch.nn.Module):
            # print('isinstance name, model', name, model)
            model.load_state_dict(state['state_dict_'+name])
    optimizer.load_state_dict(state['optimizer'])
    epoch = state["epoch"]
    # batch_idx = state["batch_idx"]
    cfg = state["cfg"] 
    best_validation_performance = state["best_validation_performance"]
    # return epoch, batch_idx, best_validation_performance
    return epoch, best_validation_performance


def load_models(models, cfg, prefix='best'):
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            fn = os.path.join(cfg['save_path'], 'checkpoints', f'{prefix}_{name}.pth')
            model.load_state_dict(torch.load(fn, map_location=cfg['device']))

def get_validation_results(dataset, models, training_pipeline, cfg):
    output = CustomSummaryTracker()
    performance = CustomSummaryTracker()
    for batch in dataset['valloader']:
        model_output = training_pipeline['forward'](batch, models, cfg, to_cpu=True)
        save_output = {key: model_output[key] for key in cfg['save_output']}
        output.update(save_output)
        loss = training_pipeline['compound_loss_func'](model_output)
        performance.update(training_pipeline['compound_loss_func'].items())
    performance = pd.DataFrame(performance.get())
    return output, performance

def save_validation_results(output, performance, cfg):
    path = os.path.join(cfg['save_path'], 'validation_results')
    #added
    if not os.path.exists(path):
        os.makedirs(path)

    print(f'Saving validation results to {path}')
    if output is not None:
        output = {k: torch.cat(v) for k, v in output.get().items()}  # concatenate batches
        save_pickle(output, path)
    performance.to_csv(os.path.join(path, 'validation_performance.csv'))
    performance.describe().to_csv(os.path.join(path, 'performance_summary.csv'))

def save_output_history(logging, cfg):
    path = os.path.join(cfg['save_path'], 'output_history')
    all_output = logging['example_output'].get()
    output = {key: val for key, val in all_output.items() if key in cfg['save_output']}
    save_pickle(output, path)


def save_training_summary(logging, cfg):
    # Write training and validation summary
    for label in ['training', 'validation']:
        fn = os.path.join(cfg['save_path'], f'{label}_summary.csv')
        data = pd.DataFrame(logging[f'{label}_summary'].get())
        data['label'] = label
        data.to_csv(fn, index=False)

def main(args):
    """"Initialize components and run training"""

    # Initialize training
    cfg = load_config(args.config)
    models = init_training.get_models(cfg)
    dataset = init_training.get_dataset(cfg)
    training_pipeline = init_training.get_training_pipeline(cfg)
    logging = init_training.get_logging(cfg)
    start = True
    train(dataset, models, training_pipeline, logging, cfg, start)
    # if cfg['pipeline'] == "unconstrained-video-reconstruction_rnn":
    #     print('train_rnn')
    #     train_rnn(dataset, models, training_pipeline, logging, cfg, start)
    # elif cfg['pipeline'] == "unconstrained-video-reconstruction":
    #     print('train non-rnn')
    #     train(dataset, models, training_pipeline, logging, cfg, start)
    #added/changed
    # train2(dataset, models, training_pipeline, logging, cfg)
    save_models(models, cfg, prefix='final')

    # Save the results
    load_models(models, cfg, prefix='best')
    save_output_history(logging, cfg)
    save_training_summary(logging, cfg)
    output, performance = get_validation_results(dataset, models, training_pipeline, cfg)
    if not args.save_output:
        output = None
    save_validation_results(output, performance, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", type=str, default=None,
                       help="filename of config file (yaml) with the training configurations: e.g. '_config.yaml' ")
    # group.add_argument("-l", "--specs-list", type=str, default=None,
    #                     help="filename of specs file (csv) with the list of model specifications") # Todo
    parser.add_argument('-s', '--save-output', action='store_true',
                        help="save the processed validation images after training")


    args = parser.parse_args()
    main(args)