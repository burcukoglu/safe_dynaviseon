import numpy as np
import os
import pickle
from glob import glob
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.datasets as ds
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageFilter 
import cv2 as cv
import string
import utils
from tqdm import tqdm

#added 
import pdb
import matplotlib.pyplot as plt
import time
from torch.profiler import profile, record_function, ProfilerActivity

def get_ade50k_dataset(cfg):
    trainset = ADE_Dataset(device=cfg['device'],
                                          directory=cfg['data_directory'],
                                          imsize=(128, 128),
                                          stimulus_length=cfg['sequence_length'], #added
                                          model = cfg['model_architecture'] , #added
                                          mode=cfg['mode'], #added
                                          load_preprocessed=cfg['load_preprocessed'])
    valset = ADE_Dataset(device=cfg['device'], directory=cfg['data_directory'],
                                        imsize=(128, 128),
                                        stimulus_length=cfg['sequence_length'], #added
                                        model = cfg['model_architecture'] , #added
                                        mode=cfg['mode'], #added
                                        load_preprocessed=cfg['load_preprocessed'],
                                        validation=True)
    return trainset, valset

def get_bouncing_mnist_dataset(cfg):
    trainset = Bouncing_MNIST(device=cfg['device'],
                                             directory=cfg['data_directory'],
                                             mode=cfg['mode'],
                                             n_frames=cfg['sequence_length'],
                                             imsize=(128, 128))
    valset = Bouncing_MNIST(device=cfg['device'],
                            directory=cfg['data_directory'],
                            mode=cfg['mode'],
                            n_frames=cfg['sequence_length'],
                            imsize=(128, 128),
                            validation=True)
    return trainset, valset


#added
def get_kitti_dataset(cfg):
    # first=time.time()
    trainset = KITTI_Dataset(device=cfg['device'],
                                             directory=cfg['data_directory'],
                                             mode=cfg['mode'],
                                             n_frames=cfg['sequence_length'],
                                             imsize=(128, 128),
                                             load_preprocessed = cfg['load_preprocessed'], 
                                             sliding_sequences=cfg['sliding_sequences'])
    valset = KITTI_Dataset(device=cfg['device'],
                            directory=cfg['data_directory'],
                            mode=cfg['mode'],
                            n_frames=cfg['sequence_length'],
                            imsize=(128, 128),
                            validation=True,
                            load_preprocessed = cfg['load_preprocessed'],
                            sliding_sequences=cfg['sliding_sequences'])

    # end=time.time()
    # print('time GET DATASET', end-first)
    # trainset, valset = KITTI_Dataset(device=cfg['device'],
    #                                          directory=cfg['data_directory'],
    #                                          mode=cfg['mode'],
    #                                          n_frames=cfg['sequence_length'],
    #                                          imsize=(128, 128))
    return trainset, valset


def create_circular_mask(h, w, center=None, radius=None, circular_mask=True):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    x = torch.arange(h)
    Y, X = torch.meshgrid(x,x)
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



class Bouncing_MNIST(Dataset):

    def __init__(self, directory='./datasets/BouncingMNIST',
                 device = torch.device('cuda:0'),
                 mode = 'recon',
                 imsize=(128,128),
                 n_frames=6,
                 validation=False,
                 circular_mask=True):
        super().__init__()
        
        VALIDATION_SPLIT = 0.1 # Fraction of sequences used as validation set

        self.device = device
        self.mode = mode
        self.imsize = imsize
        self.n_frames = n_frames #seq_length
        full_set = np.load(directory+'mnist_test_seq.npy').transpose(1, 0, 2, 3) # -> (Batch, Frame, Height, Width)
        print('fillset',full_set.shape ) #(10000, 20, 64, 64)
        
        n_val = int(VALIDATION_SPLIT*full_set.shape[0])
        print('nval',n_val)
        if validation:
            data = torch.from_numpy(full_set[:n_val])
            print('valid', data.shape)
        else:
            data = torch.from_numpy(full_set[n_val:])
            print('train', data.shape)

        
        _, seq_len, H, W  = data.shape # (original sequence has 20 frames)
        
        # In reconstruction mode, the target is same as input and only one set of images is returned
        if self.mode=='recon':
            
            # Use the remaining frames if the original sequence length (20) fits multiple output sequences (n_frames)
            divisor = seq_len//n_frames 
            print('divisor',divisor)
            full_set = data[:,:n_frames*divisor]
            print('fullset', full_set.shape)
            if divisor>1: 
                data = data.reshape((-1,n_frames,H,W))
                print('fullset reshape', data.shape)

        self.data = data.unsqueeze(dim=1) # Add (grayscale) channel 
        print('selfdata',self.data.shape) #torch.Size([4000, 1, 5, 64, 64])
                    
        if circular_mask:
            self._mask = create_circular_mask(*imsize).repeat(1,n_frames,1,1) #(Channel, Frame, Height, Width)
        else:
            self._mask = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.mode == 'recon':
            frames = T.Resize(128)(self.data[i]/255.)
            if self._mask is not None:
                frames = frames*self._mask
            print('inputted frames',frames.shape) #torch.Size([1, 5, 128, 128])
            return frames.detach().to(self.device)
        elif self.mode == 'recon_pred':
            input_frames = T.Resize(128)(self.data[i,:,:self.n_frames]/255.)#.to(self.device)
            future_frames = T.Resize(128)(self.data[i,:,self.n_frames:self.n_frames*2]/255.)#.to(self.device)
            
            if self._mask is not None:
                input_frames = input_frames*self._mask
                future_frames = future_frames*self._mask
            print('input_frames',input_frames.shape) 
            print('future_frames',future_frames.shape) 
            return input_frames.detach().to(self.device), future_frames.detach().to(self.device)
            
class ADE_Dataset(Dataset):
    
    def __init__(self, directory='../_Datasets/ADE20K/',
                 device=torch.device('cuda:5'),
                 imsize = (128,128),
                 stimulus_length = 1,#added
                 model = None , #added
                 mode = 'recon', #added
                 grayscale = True,
                 normalize = True,
                 contour_labels = True,
                 validation=False,
                 load_preprocessed=False,
                 circular_mask=True):
        
        self.validation = validation
        self.contour_labels = contour_labels
        self.normalize = normalize
        self.grayscale = grayscale
        self.device = device

        #added/todoq
        self.stimulus_length = stimulus_length
        self.model = model
        self.mode = mode
    
        contour = lambda im: im.filter(ImageFilter.FIND_EDGES).point(lambda p: p > 1 and 255) if self.contour_labels else im
        # to_grayscale = lambda im: im.convert('L') if self.grayscale else im

        # Image and target tranformations (square crop and resize)
        self.img_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                        T.Resize(imsize),
                                        T.ToTensor()
                                        ])
        self.trg_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                        T.Resize(imsize,interpolation=T.InterpolationMode.NEAREST), # Nearest Neighbour
                                        T.Lambda(contour),
                                        T.ToTensor()
                                        ])
        
        # Normalize
        self.normalizer = T.Normalize(mean = [0.485, 0.456, 0.406],
                                      std = [0.229, 0.224, 0.225])        
        
        if circular_mask:
            self._mask = create_circular_mask(*imsize).view(1,*imsize)
        else:
            self._mask = None
        
        # RGB converter
        weights=[.3,.59,.11]
        self.to_grayscale = lambda image:torch.sum(torch.stack([weights[c]*image[c,:,:] for c in range(3)],dim=0),
                                                   dim=0,
                                                   keepdim=True)

        self.inputs = []
        self.targets = []
        
        
        if load_preprocessed:
            self.load(directory)
#             print('----Loaded preprocessed data----')
#             print(f'input length: {len(self.inputs)} samples')
        else:
            # Collect files 
            img_files, seg_files = [],[]
            print('----Listing training images----')
            for path, subdirs, files in tqdm(os.walk(os.path.join(directory,'images','training'))):
            # for path, subdirs, files in os.walk(os.path.join(directory,'training')):
                img_files+= glob(os.path.join(path,'*.jpg'))
                seg_files+= glob(os.path.join(path,'*seg.png'))
                val_img_files, val_seg_files, = [],[]
            print('----Listing validation images----')
            for path, subdirs, files in tqdm(os.walk(os.path.join(directory,'images','validation'))):
            # for path, subdirs, files in os.walk(os.path.join(directory,'validation')):
                val_img_files+= glob(os.path.join(path,'*.jpg'))
                val_seg_files+= glob(os.path.join(path,'*seg.png'))
            for l in [img_files,seg_files,val_img_files,val_seg_files]:
                l.sort()

            print('Finished listing files')
            # Image and target files
            if validation:
                self.input_files = val_img_files
                self.target_files = val_seg_files
            else:
                self.input_files = img_files
                self.target_files = seg_files

            print('----Preprocessing ADE20K input----')
            for image, target in tqdm(zip(self.input_files, self.target_files),total=len(self.input_files)):
                im = Image.open(image).convert('RGB')
                t = Image.open(target).convert('L')

                # Crop, resize & transform
                x = self.img_transform(im)
                t = self.trg_transform(t)
                                            
                # Additional tranforms:
                if self.normalize:
                    x = self.normalizer(x)
                if self.grayscale:
                    x = self.to_grayscale(x)

                self.inputs += [x]
                self.targets += [t]
            print('----Finished preprocessing----')
            self.save(directory)

    def save(self,directory):
        
        # Make directory if it doesn't exist
        path = os.path.join(directory, 'processed')
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save files
        mode = '_val' if self.validation else '_train'
        with open(os.path.join(path,f'standardized_processed{mode}_inputs.pkl'),'wb') as f:
            pickle.dump(self.inputs,f)
        with open(os.path.join(path,f'standardized_processed{mode}_targets.pkl'),'wb') as f:
            pickle.dump(self.targets,f)

    def load(self,directory):
        mode = '_val' if self.validation else '_train'
        with open(os.path.join(directory,f'processed{mode}','_inputs.pkl'),'rb') as f:
        # with open(os.path.join(directory,'processed',f'standardized_processed{mode}_inputs.pkl'),'rb') as f:
            self.inputs = pickle.load(f)
        with open(os.path.join(directory,f'processed{mode}','_targets.pkl'),'rb') as f:
        # with open(os.path.join(directory,'processed',f'standardized_processed{mode}_targets.pkl'),'rb') as f:
            self.targets = pickle.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        
        x = self.inputs[i]
        # t = self.targets[i]
        #added/changed
        if self.mode == 'recon_label':
            t = self.targets[i]
        

        if self._mask is not None:
            x = x*self._mask
            # t = t*self._mask
            #added/changed
            if self.mode == 'recon_label':
                t = t*self._mask
    
        #added/todo
        if self.stimulus_length >= 1: #for zhao #change = if dealing with image data!
            # x = x.repeat(1,self.stimulus_length,1,1) #for n,c,d,h,w,
            # t = t.repeat(1,self.stimulus_length,1,1)
            if 'zhao-autoencoder' in self.model:
                x = x.repeat(1,self.stimulus_length,1,1) #for n,c,d,h,w, shape c,d,h,w
                # t = t.repeat(1,self.stimulus_length,1,1)
                #added/changed
                if self.mode == 'recon_label':
                    t = t.repeat(1,self.stimulus_length,1,1)
                # for i in range(self.stimulus_length):
                #     plt.imsave(f'/home/burkuc/data/static/v_img_{i}.png', x[0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
                #     plt.imsave(f'/home/burkuc/data/static/v_lbl_{i}.png', t[0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
            # elif self.model == 'end-to-end-autoencoder': #or 'end-to-end-autoencoder2' :
            else:
                x = x.repeat(self.stimulus_length,1,1) #for n,c,h,w, shape ,c,h,w
                # t = t.repeat(self.stimulus_length,1,1)
                if self.mode == 'recon_label':
                    t = t.repeat(self.stimulus_length,1,1)
                # for i in range(self.stimulus_length):
                    # plt.imsave(f'/home/burkuc/data/static/c_img_{i}.png', x[i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
                    # plt.imsave(f'/home/burkuc/data/static/c_lbl_{i}.png', t[i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        # pdb.set_trace()
        # pdb.disable()
        # print('shapes', x.shape, t.shape, x.dtype, t.dtype)
        if self.mode == 'recon':
            return x.detach().to(self.device)
        return x.detach().to(self.device),t.detach().to(self.device)
    
    
class Character_Dataset(Dataset):
    """ Pytorch dataset containing images of single (synthetic) characters.
    __getitem__ returns an image containing one of 26 ascci lowercase characters, 
    typed in one of 47 fonts(default: 38 train, 9 validation) and the corresponding
    alphabetic index as label.
    """
    def __init__(self,directory = './datasets/Characters/',
                 device=torch.device('cuda:0'),
                 imsize = (128,128),
                 train_val_split = 0.8,
                 validation=False,
                 word_scale=.8,
                 invert = True, 
                 circular_mask=True): 
        
        self.imsize = imsize
        self.tensormaker = T.ToTensor()
        self.device = device
        self.validation = validation
        self.word_scale = word_scale
        self.invert = invert
        
        if circular_mask:
            self._mask = create_circular_mask(*imsize).view(1,*imsize)
        else:
            self._mask = None
        
        
        
        characters = string.ascii_lowercase
        fonts = glob(os.path.join(directory,'Fonts/*.ttf'))
        
        self.split = round(len(fonts)*train_val_split)
        train_data, val_data = [],[]
        for c in characters:
            for f in fonts[:self.split]:
                train_data.append((f,c))
            for f in fonts[self.split:]:
                val_data.append((f,c))
        self.data = val_data if validation else train_data
        self.classes = characters
        self.lookupletter = {letter: torch.tensor(index) for index, letter in enumerate(characters)}
        self.padding_correction = 6 #By default, PILs ImageDraw function uses excessive padding                                                          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        # Load font and character
        f,c = self.data[i]
        
        # Get label (alphabetic index of character)
        lbl  = self.lookupletter[c]
        
        # Scale character to image
        fontsize = 1
        font = ImageFont.truetype(f,fontsize)
        while max(font.getsize(c))/min(self.imsize) <= self.word_scale:
            fontsize += 1
            font = ImageFont.truetype(f,fontsize)
        fontsize -=1 

        # Calculate left-over space 
        font = ImageFont.truetype(f,fontsize)
        textsize = font.getsize(c)
        free_space = np.subtract(self.imsize,textsize)
        free_space += self.padding_correction

        # Draw character at random position
        img = Image.fromarray(255*np.ones(self.imsize).astype('uint8'))
        draw = ImageDraw.Draw(img)
        location = np.random.rand(2)*(free_space)
        location[1]-= self.padding_correction
        draw.text(location,c,(0,),font=font)       
        img = self.tensormaker(img)
        
        
        if self.invert:
            img = 1-img
            
        if self._mask is not None:
            img = img*self._mask

        return img.to(self.device), lbl.to(self.device)

#added
class KITTI_Dataset(Dataset):

    def __init__(self, directory='./datasets/kitti',
                 device = torch.device('cuda:0'),
                 mode = 'recon',
                 imsize=(128,128),
                 n_frames=10, #78,
                 validation=False,
                 circular_mask=True,
                 load_preprocessed=False,
                 load_equal=False,
                 sliding_sequences=False):
        super().__init__()
        
        TRAINING_SPLIT = 0.9 # Fraction of sequences used as validation set

        self.device = device
        self.mode = mode
        self.imsize = imsize
        self.n_frames = n_frames #seq_length
        self.n_chunks = 10
        # full_set = np.load(directory+'mnist_test_seq.npy').transpose(1, 0, 2, 3) # -> (Batch, Frame, Height, Width)
        
        self.img_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))), #375, 375
                                        T.Resize(imsize), #to desired size
                                        T.ToTensor() #automatic range within [0,1]
                                        ])
        # print('fillset',full_set.shape ) #(10000, 20, 64, 64)
        
        # #added
        # self.input_files = []
        # self.data_videos = [] 
        self.directory = directory
        self.validation = validation

        self.step_size = 2
        self.gpu_last = False
        print('self.gpu_last', self.gpu_last, flush=True)
        
        # if not load_equal:
        if load_preprocessed: #gave up because of divisor issue..
            print('loading')
            # self.load(directory+'processed/')
            data_videos = self.load(directory+'processed/')
            # NAS: Immediately put your data videos on to the device
            data_videos = [d.to(self.device) for d in data_videos]
            print('loaded')

        else:

            # Collect files 
            img_sequences = {}
            # img_files = []
            print('----Listing training images----')
            for path, subdirs, files in tqdm(os.walk(os.path.join(directory,'training','image_02'))):
                for seq_path in os.listdir(path):
                    if os.path.isdir(os.path.join(path,seq_path)):
                    # print('sp', seq_path)
                        img_files = []
                        img_files+= glob(os.path.join(path,seq_path,'*.png'))
                        for l in [img_files]:
                            l.sort()
                        img_sequences['tr_'+seq_path] = img_files
            print('sequences',img_sequences)       
            print('----Listing validation images----')
            for path, subdirs, files in tqdm(os.walk(os.path.join(directory,'testing','image_02'))):
                for seq_path in os.listdir(path):
                    if os.path.isdir(os.path.join(path,seq_path)):
                        val_img_files = []
                        val_img_files+= glob(os.path.join(path, seq_path,'*.png'))
                        for l in [val_img_files]:
                            l.sort()
                        # print('valimgf', val_img_files)
                        img_sequences['val_'+seq_path] = val_img_files
            print('sequences',img_sequences) 
            print('Finished listing files')
                

            # self.data_videos = []
            data_videos = []
            for key, value in img_sequences.items():
                # print(key,len(value))

                # divisor = len(value)//n_frames 

                # print('divisor',divisor)
                data_images = []
                for img in value:
                    im = Image.open(img).convert('L') #(375, 1242) , (374, 1238) #shall i take more videos out of this? but then it ruins the centeral point of view.
                    # print('imshape', np.asarray(im).shape)
                    x = self.img_transform(im) #check
                    data_images.append(x)
                    # print('data', x.shape) #torch.Size([1, 128, 128])
                    # plt.imsave(f'/home/burkuc/data/static/kitti.png', x[0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
                data_images = torch.stack(data_images, dim=1)

                # self.data_videos.append(data_images)
                data_videos.append(data_images)
            # self.save(directory)
            self.save(directory, data_videos)

        equal_sequences =[]

        if sliding_sequences: #with each frame starts a new video, thus they have an overlap (used to create more video data)
            # for video in self.data_videos:
            for video in data_videos:
                # for i in range(video.shape[1]-n_frames+1):
                for i in range(0, video.shape[1]-n_frames+1, self.step_size):  # step=2 determines how much overlap there is
                    sequence = video[:,i:i+n_frames]
                    # print('seq', sequence.shape)
                    equal_sequences.append(sequence)
                

        else: #videos don't have overlaps

            # for video in self.data_videos:
            for video in data_videos:
                # print('video',video.shape,video.shape[1])
                divisor = video.shape[1]//n_frames 
                # for i in range(video.shape[1]):
                    # pdb.set_trace()
                    # pdb.enable()
                    
                #     plt.imsave(f'/home/burkuc/data/static/0kitti4{i}.png', video[0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
                # pdb.set_trace()
                # pdb.enable()
                full_set_images = video[:,:n_frames*divisor] 
                _, _, H, W  = full_set_images.shape #torch.Size([1, 110, 128, 128])
                # print('divisor', full_set_images.shape )
                if divisor>1: 
                    full_set_images = full_set_images.reshape((-1,n_frames,H,W)) 
                # print('divisor2', full_set_images.shape )
                equal_sequences.append(full_set_images)

        # self.input_files = torch.cat(equal_sequences, dim=0).unsqueeze(dim=1)
        input_files = torch.cat(equal_sequences, dim=0).unsqueeze(dim=1)
        # print('input', self.input_files.shape) #torch.Size([1889, 1, 10, 128, 128]) #torch.Size([50, 1, 0, 128, 128]) torch.Size([18653, 1, 10, 128, 128]) torch.Size([1866, 1, 10, 128, 128])    
        print('input', input_files.shape) #orch.Size([9339, 1, 10, 128, 128])

        # #added
        # self.save_equal(directory, input_files)
        # if not load_equal:
        # n_tr = int(TRAINING_SPLIT*self.input_files.shape[0])
        n_tr = int(TRAINING_SPLIT*input_files.shape[0])
        # print('ntr',n_tr,'nval', self.input_files.shape[0]-n_tr)
        print('ntr',n_tr,'nval', input_files.shape[0]-n_tr)
        if validation:
            # self.data = self.input_files[n_tr:]
            # self.data = input_files[n_tr:]#.to(self.device) #UNCOMMENT
            data = input_files[n_tr:]#
            # print('valid', self.data.shape)
            print('valid', data.shape)
        else:
            # self.data = self.input_files[:n_tr]
            # self.data = input_files[:n_tr]#.to(self.device) #UNCOMMENT
            data = input_files[:n_tr]
            # print('train', self.data.shape)
            print('train', data.shape)
            
        # EQUAL DATA NO LONGER BEING USED
        # self.save_equal(directory, data, self.n_frames, self.step_size)
        
        # else: # IF LOAD EQUAL:
        #     if self.gpu_last:
        #         pass
        #     else:
        #         data = self.load_equal(self.directory+'equalized/', self.n_frames, self.step_size) 
        #         print('data', data.shape, 'val:', self.validation)

        if circular_mask:
            self._mask = create_circular_mask(*imsize).repeat(1,n_frames,1,1)
            # self._mask = create_circular_mask(*imsize).repeat(1,n_frames,1,1).to(self.device) #(Channel, Frame, Height, Width)
        else:
            self._mask = None
            
        # #added
        # self.save(directory)
        #added
        # if load_equal: 
        #     if self.gpu_last:
        #         pass
        #     else:
        #         data = self.load_equal(self.directory+'equalized/', self.n_frames, self.step_size) 
        #         print('data', data.shape, 'val:', self.validation)
        # else:
        #     self.save_equal(directory, data, self.n_frames, self.step_size)

        #UNCOMMENT IF TO GPU ALL AT ONCE
        if self._mask is not None:
            self.data = data*self._mask.to(self.device)

        # #UNCOMMENT IF TO GPU ALL AT ONCE
        # if not self.gpu_last:
        #     self.data = self.data.to(device)
        # print('datadevice', self.data.device, self.data.shape)
        # print('data',self.data)
    
    # def save(self,directory):
    def save(self,directory, data_videos):
        
        # Make directory if it doesn't exist
        path = os.path.join(directory, 'processed')
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save files
        # mode = '_val' if self.validation else '_train'
        with open(os.path.join(path,f'processed_inputs.pkl'),'wb') as f:
            # pickle.dump(self.data_videos,f)
            pickle.dump(data_videos,f)

    def load(self,directory):
        # mode = '_val' if self.validation else '_train'
        with open(os.path.join(directory,f'processed_inputs.pkl'),'rb') as f:
        # with open(os.path.join(directory,'processed',f'standardized_processed{mode}_inputs.pkl'),'rb') as f:
            # self.inputs = pickle.load(f)
            # self.data_videos = pickle.load(f)
            data_videos = pickle.load(f)
            return data_videos

    #added
    # def save_equal(self,directory, data_videos, n_frames, step_size):
        
    #     # Make directory if it doesn't exist
    #     path = os.path.join(directory, 'equalized')
    #     if not os.path.exists(path):
    #         os.makedirs(path)
        
    #     # Save files
    #     mode = '_val' if self.validation else '_train'
    #     chunk_sizes = int(np.ceil(len(data_videos)/self.n_chunks))
    #     for c in range(self.n_chunks):
    #         print(c, flush=True)
    #         with open(os.path.join(path,f'equal_{n_frames}{mode}_{step_size}_inputs_chunk{c}.pkl'),'wb') as f:
    #             # pickle.dump(self.data_videos,f)
    #             pickle.dump(data_videos[c*chunk_sizes:(c+1)*chunk_sizes],f)

    # def load_equal(self,directory, n_frames, step_size):
    #     mode = '_val' if self.validation else '_train'
    #     chunks = []
    #     for c in self.n_chunks:
    #         with open(os.path.join(directory,f'equal_{n_frames}{mode}_{step_size}_inputs_chunk{c}.pkl'),'rb') as f:
    #         # with open(os.path.join(directory,'processed',f'standardized_processed{mode}_inputs.pkl'),'rb') as f:
    #             # self.inputs = pickle.load(f)
    #             # self.data_videos = pickle.load(f)
    #             e = pickle.load(f)
    #         e = e.to(self.device)
    #         chunks.append(e)
    #     return torch.stack(chunks)
    #     # return equal_data_videos

    def __len__(self):
        return len(self.data)
        # if self.gpu_last:
        #     return len(self.load_equal(self.directory+'equalized/', self.n_frames, self.step_size))
        # else:
        #     return len(self.data)
        # return len(self.input_files)

    def __getitem__(self, i):
        if self.mode == 'recon':
            
                # model(inputs)
        
            # frames = T.Resize(128)(self.data[i]/255.)
            # first=time.time()
            # if self.gpu_last:
            #     data = self.load_equal(self.directory+'equalized/', self.n_frames, self.step_size) #but slower if one by one
            # # print('data',data.shape, 'i', i)
            #     frames = data[i]
            # else:
            #     frames = self.data[i]
            frames = self.data[i]

            # frames = self.input_fles[i]
            # for i in range(frames.shape[1]):
            #     plt.imsave(f'/home/burkuc/data/static/kitti3{i}.png', frames[0,i,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
            # print('frames', frames.shape, frames.min(), frames.max())

            #COMMENT IF TO GPU ALL AT ONCE
            # if self.gpu_last:
            #     if self._mask is not None:
            #         frames = frames*self._mask
            # end= time.time()
            # print('getitem', end-first)
            # print('inputted frames',frames.shape) #torch.Size([1, 5, 128, 128])
            # return frames.detach().to(self.device)
                
            # if self.gpu_last:
            #         return frames.detach().to(self.device) #UNCOMMENT IF TO GPU DIRECTLY IN INIT
            # else:
            #     return frames.detach()
            return frames.detach()
            
        elif self.mode == 'recon_pred':
            input_frames = T.Resize(128)(self.data[i,:,:self.n_frames]/255.)#.to(self.device)
            future_frames = T.Resize(128)(self.data[i,:,self.n_frames:self.n_frames*2]/255.)#.to(self.device)
            
            # if self._mask is not None: #redo this pay attention to where mask is created
            #     input_frames = input_frames*self._mask
            #     future_frames = future_frames*self._mask
            print('input_frames',input_frames.shape) 
            print('future_frames',future_frames.shape) 
            # return input_frames.detach().to(self.device), future_frames.detach().to(self.device)
            return input_frames.detach(), future_frames.detach()
            