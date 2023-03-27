import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

#added
from torch.cuda.amp import custom_bwd, custom_fwd

def get_e2e_autoencoder(cfg):

    # initialize encoder and decoder
    if cfg['output_steps'] is not None:
        encoder = torch.nn.Sequential(E2E_Encoder(in_channels=cfg['in_channels'],
                                                  out_channels=cfg['out_channels'], #added
                                                        n_electrodes=cfg['n_electrodes'],
                                                        out_scaling=1.,
                                                        out_activation='sigmoid'),
                                      SafetyLayer(n_steps=10,
                                                        order=2,
                                                        out_scaling=cfg['output_scaling'])).to(cfg['device'])
    else:
        encoder = E2E_Encoder(in_channels=cfg['in_channels'], #added/changed from in_channels=1,
                                    out_channels=cfg['out_channels'], #added
                                    n_electrodes=cfg['n_electrodes'],
                                    out_scaling=cfg['output_scaling'],
                                    out_activation='relu').to(cfg['device'])

    decoder = E2E_Decoder(out_channels=cfg['out_channels'],
                          in_channels=cfg['in_channels'], #added
                          out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

#added
def get_e2e_recurrent_net(cfg):

    # initialize encoder and decoder
    # if cfg['output_steps'] is not None:
    #     encoder = torch.nn.Sequential(E2E_Encoder(in_channels=cfg['in_channels'],
    #                                               out_channels=cfg['out_channels'], #added
    #                                                     n_electrodes=cfg['n_electrodes'],
    #                                                     out_scaling=1.,
    #                                                     out_activation=cfg['out_activation']), #'sigmoid'),
    #                                   SafetyLayer(n_steps=10,
    #                                                     order=2,
    #                                                     out_scaling=cfg['output_scaling'])).to(cfg['device'])
    # else:
    encoder = E2E_Encoder_RNN(in_channels=cfg['in_channels'], #added/changed from in_channels=1,
                                    out_channels=cfg['sequence_length'], # cfg['out_channels'], #added
                                    n_electrodes=cfg['n_electrodes'],
                                    out_scaling=cfg['output_scaling'],
                                    # out_activation='relu').to(cfg['device'])
                                    out_activation = cfg['out_activation'],#).to(cfg['device']
                                    hidden_size = cfg['rnn_hidden_size'], # needs to be same
                                    num_layers = cfg["rnn_num_layers"],
                                    rnn_input_size=cfg['rnn_input_size'],
                                    constrained=cfg['constrained']).to(cfg['device'])

    decoder = E2E_Decoder(out_channels=cfg['sequence_length'], #cfg['out_channels'],
                          in_channels=cfg['in_channels'], #added
                          out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

#added
def get_e2e_recurrent_net_out3(cfg):

    # initialize encoder and decoder
    # if cfg['output_steps'] is not None:
    #     encoder = torch.nn.Sequential(E2E_Encoder(in_channels=cfg['in_channels'],
    #                                               out_channels=cfg['out_channels'], #added
    #                                                     n_electrodes=cfg['n_electrodes'],
    #                                                     out_scaling=1.,
    #                                                     out_activation=cfg['out_activation']), #'sigmoid'),
    #                                   SafetyLayer(n_steps=10,
    #                                                     order=2,
    #                                                     out_scaling=cfg['output_scaling'])).to(cfg['device'])
    # else:
    encoder = E2E_Encoder_RNN_out3(in_channels=cfg['in_channels'], #added/changed from in_channels=1,
                                    out_channels=cfg['sequence_length'], # cfg['out_channels'], #added
                                    n_electrodes=cfg['n_electrodes'],
                                    out_scaling=cfg['output_scaling'],
                                    # out_activation='relu').to(cfg['device'])
                                    out_activation = cfg['out_activation'],#).to(cfg['device']
                                    hidden_size = cfg['rnn_hidden_size'], # needs to be same
                                    num_layers = cfg["rnn_num_layers"],
                                    rnn_input_size=cfg['rnn_input_size'],
                                    constrained=cfg['constrained']).to(cfg['device'])

    decoder = E2E_Decoder(out_channels=cfg['sequence_length'], #cfg['out_channels'],
                          in_channels=cfg['in_channels'], #added
                          out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

#added
def get_e2e_recurrent_net_out32(cfg):

    # initialize encoder and decoder
    # if cfg['output_steps'] is not None:
    #     encoder = torch.nn.Sequential(E2E_Encoder(in_channels=cfg['in_channels'],
    #                                               out_channels=cfg['out_channels'], #added
    #                                                     n_electrodes=cfg['n_electrodes'],
    #                                                     out_scaling=1.,
    #                                                     out_activation=cfg['out_activation']), #'sigmoid'),
    #                                   SafetyLayer(n_steps=10,
    #                                                     order=2,
    #                                                     out_scaling=cfg['output_scaling'])).to(cfg['device'])
    # else:
    encoder = E2E_Encoder_RNN_out32(in_channels=cfg['in_channels'], #added/changed from in_channels=1,
                                    out_channels=cfg['sequence_length'], # cfg['out_channels'], #added
                                    n_electrodes=cfg['n_electrodes'],
                                    out_scaling=cfg['output_scaling'],
                                    # out_activation='relu').to(cfg['device'])
                                    out_activation = cfg['out_activation'],#).to(cfg['device']
                                    hidden_size = cfg['rnn_hidden_size'], # needs to be same
                                    num_layers = cfg["rnn_num_layers"],
                                    rnn_input_size=cfg['rnn_input_size'],
                                    constrained=cfg['constrained'],
                                    constraint_coeff=cfg['constraint_coeff']).to(cfg['device'])

    decoder = E2E_Decoder(out_channels=cfg['sequence_length'], #cfg['out_channels'],
                          in_channels=cfg['in_channels'], #added
                          out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

#added
def get_e2e_autoencoder2(cfg):

    # initialize encoder and decoder
    if cfg['output_steps'] is not None:
        encoder = torch.nn.Sequential(E2E_Encoder2(in_channels=cfg['in_channels'],
                                                  out_channels=cfg['out_channels'], #added
                                                        n_electrodes=cfg['n_electrodes'],
                                                        out_scaling=1.,
                                                        out_activation='sigmoid'),
                                      SafetyLayer(n_steps=10,
                                                        order=2,
                                                        out_scaling=cfg['output_scaling'])).to(cfg['device'])
    else:
        encoder = E2E_Encoder2(in_channels=cfg['in_channels'], #added/changed from in_channels=1,
                                    out_channels=cfg['out_channels'], #added
                                    n_electrodes=cfg['n_electrodes'],
                                    out_scaling=cfg['output_scaling'],
                                    out_activation='relu').to(cfg['device'])

    decoder = E2E_Decoder(out_channels=cfg['out_channels'],
                          in_channels=cfg['in_channels'], #added
                          out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

def get_Zhao_autoencoder(cfg):
    #encoder = ZhaoEncoder(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    #changed/added
    encoder = ZhaoEncoder(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes'], out_scaling=cfg['output_scaling'], constrained=cfg['constrained']).to(cfg['device'])
    # encoder = ZhaoEncoder_clamp(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes'], out_scaling=cfg['output_scaling']).to(cfg['device'])
    decoder = ZhaoDecoder(out_channels=cfg['out_channels'], out_activation=cfg['out_activation']).to(cfg['device'])
    #changed/added
    # encoder = ZhaoEncoder(in_channels=cfg['in_channels'], out_channels=cfg['out_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    # decoder = ZhaoDecoder(out_channels=cfg['out_channels'], in_channels=cfg['in_channels'], out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

#added
def get_Zhao_autoencoder2(cfg):
    # encoder = ZhaoEncoder(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    #changed/added
    encoder = ZhaoEncoder2(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes'], sequence_length=cfg['sequence_length'], output_scaling=cfg['output_scaling']).to(cfg['device'])
    decoder = ZhaoDecoder(out_channels=cfg['out_channels'], out_activation=cfg['out_activation']).to(cfg['device'])
    #changed/added
    # encoder = ZhaoEncoder(in_channels=cfg['in_channels'], out_channels=cfg['out_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    # decoder = ZhaoDecoder(out_channels=cfg['out_channels'], in_channels=cfg['in_channels'], out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

def get_Zhao_autoencoder_out3(cfg): 
    #encoder = ZhaoEncoder(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    #changed/added
    encoder = ZhaoEncoder_out3(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes'], out_scaling=cfg['output_scaling'], constrained=cfg['constrained']).to(cfg['device'])
    decoder = ZhaoDecoder(out_channels=cfg['out_channels'], out_activation=cfg['out_activation']).to(cfg['device'])
    #changed/added
    # encoder = ZhaoEncoder(in_channels=cfg['in_channels'], out_channels=cfg['out_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    # decoder = ZhaoDecoder(out_channels=cfg['out_channels'], in_channels=cfg['in_channels'], out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

#added
def get_Zhao_autoencoder_out32(cfg): 
    #encoder = ZhaoEncoder(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    #changed/added
    encoder = ZhaoEncoder_out32(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes'], out_scaling=cfg['output_scaling'], constrained=cfg['constrained'], constrained_based_on=cfg['constrained_based_on'], constraint_coeff=cfg['constraint_coeff']).to(cfg['device'])
    decoder = ZhaoDecoder(out_channels=cfg['out_channels'], out_activation=cfg['out_activation']).to(cfg['device'])
    #changed/added
    # encoder = ZhaoEncoder(in_channels=cfg['in_channels'], out_channels=cfg['out_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    # decoder = ZhaoDecoder(out_channels=cfg['out_channels'], in_channels=cfg['in_channels'], out_activation=cfg['out_activation']).to(cfg['device'])

    return encoder, decoder

def convlayer(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer


def convlayer3d(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=False), #True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer 

def deconvlayer3d(n_input, n_output, k_size=2, stride=2, padding=0, dilation=1, resample_out=None):
    layer = [
        nn.ConvTranspose3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=False), #True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out
    
  
class SafetyLayer(torch.nn.Module):
    def __init__(self, n_steps=5, order=1, out_scaling=120e-6):
        super(SafetyLayer, self).__init__()
        self.n_steps = n_steps
        self.order = order
        self.output_scaling = out_scaling

    def stairs(self, x):
        """Assumes input x in range [0,1]. Returns quantized output over range [0,1] with n quantization levels"""
        return torch.round((self.n_steps-1)*x)/(self.n_steps-1)

    def softstairs(self, x):
        """Assumes input x in range [0,1]. Returns sin(x) + x (soft staircase), scaled to range [0,1].
        param n: number of phases (soft quantization levels)
        param order: number of recursion levels (determining the steepnes of the soft quantization)"""

        return (torch.sin(((self.n_steps - 1) * x - 0.5) * 2 * math.pi) +
                         (self.n_steps - 1) * x * 2 * math.pi) / ((self.n_steps - 1) * 2 * math.pi)
    
    def forward(self, x):
        out = self.softstairs(x) + self.stairs(x).detach() - self.softstairs(x).detach()
        return (out * self.output_scaling).clamp(1e-32,None)


class VGGFeatureExtractor():
    def __init__(self,layer_names=['1','3','6','8'], layer_depth=9 ,device='cpu'):
        
        # Load the VGG16 model
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = torch.nn.Sequential(*[*model.features][:layer_depth]).to(device)
        
        # Register a forward hook for each layer of interest
        self.layers = {name: layer for name, layer in self.feature_extractor.named_children() if name in layer_names}
        self.outputs = dict()
        for name, layer in self.layers.items():
            layer.__name__ = name
            layer.register_forward_hook(self.store_output)
            
    def store_output(self, layer, input, output):
        self.outputs[layer.__name__] = output

    def __call__(self, x):
        
        # If grayscale, convert to RGB
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        
        # Forward pass
        self.feature_extractor(x)
        activations = list(self.outputs.values())
        
        return activations
        
        

class E2E_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu'):
        super(E2E_Encoder, self).__init__()
        self.output_scaling = out_scaling
        self.out_activation = {'tanh': nn.Tanh(), ## NOTE: simulator expects only positive stimulation values 
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                   *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   *convlayer(32,16,3,1,1),
                                   nn.Conv2d(16,1,3,1,1),
                                   nn.Flatten(),
                                   nn.Linear(1024,n_electrodes),
                                   self.out_activation)

    def forward(self, x):
        self.out = self.model(x)
        stimulation = self.out*self.output_scaling #scaling improves numerical stability
        return stimulation

class E2E_Encoder2(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu'):
        super(E2E_Encoder2, self).__init__()
        self.output_scaling = out_scaling
        self.out_activation = {'tanh': nn.Tanh(), ## NOTE: simulator expects only positive stimulation values 
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                   *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   *convlayer(32,16,3,1,1),
                                   nn.Conv2d(16,out_channels,3,1,1), #nn.Conv2d(16,1,3,1,1),
                                   nn.Flatten(start_dim=2), #nn.Flatten(),
                                   nn.Linear(1024,n_electrodes),
                                   self.out_activation)

    def forward(self, x):
        self.out = self.model(x)
        stimulation = self.out*self.output_scaling #scaling improves numerical stability
        return stimulation

#added
class E2E_Encoder_RNN(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', hidden_size=512, num_layers=1, rnn_input_size=2000, constrained=True):
        super(E2E_Encoder_RNN, self).__init__()
        self.output_scaling = out_scaling
        # self.out_activation = {'tanh': nn.Tanh(), ## NOTE: simulator expects only positive stimulation values 
        #                        'sigmoid': nn.Sigmoid(),
        #                        'relu': nn.ReLU(),
        #                        'softmax':nn.Softmax(dim=1)}[out_activation]
        
        self.constrain = constrained

        # Model
        # self.enc = ResNetEncoder()

        self.enc = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                #    *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(8,16,3,2,1,resample_out=None),
                                #    *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,2,1,resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                #    ResidualBlock(32, resample_out=None),
                                #    ResidualBlock(32, resample_out=None),
                                   *convlayer(32,out_channels,3,1,1)) #, #*convlayer(32,16,3,1,1),
                                #    nn.Conv2d(16,out_channels,3,1,1)) #, #nn.Conv2d(16,1,3,1,1),
                                #   nn.Flatten(start_dim=2), #nn.Flatten(),
                                #    nn.Linear(1024,n_electrodes),
                                #    self.out_activation)

        self.flatten = nn.Flatten(start_dim=2)
        self.dense = nn.Linear(1024,rnn_input_size)
        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=num_layers)
        # self.rnn = nn.LSTM(input_size=512, hidden_size=512, num_layers=num_layers)
        self.readout = nn.Linear(hidden_size, n_electrodes)
        # frames torch.Size([2, 10, 128, 128])
        # encoded torch.Size([2, 10, 32, 32])
        # flattened torch.Size([2, 10, 1024]) #does it make sense to g(o from 1024 to 2000?

        if self.constrain:
            self.out_activation=nn.Sigmoid()
        else:
            self.out_activation=nn.ReLU()

        # old:
        # dense torch.Size([2, 10, 2000])
        # permute torch.Size([10, 2, 2000]) #does it make sense to go from 2000 to 1000 via rnn?

        # new:
        # dense torch.Size([2, 10, 512])
        # permute torch.Size([10, 2, 512])

        # rnn_out torch.Size([10, 2, 1000])
        # h torch.Size([1, 2, 1000])
        # c torch.Size([1, 2, 1000])
        # stimulation torch.Size([10, 2, 1000])

    def forward(self, x, hidden_state):
        h = hidden_state[0]
        c = hidden_state[1]
        out = self.enc(x)
        # print('encoded', self.out.shape)
        out= self.flatten(out)
        # print('flattened', self.out.shape)
        out = self.dense(out)
        # print('dense', self.out.shape)
        out= out.permute(1,0,2)
        # print('permute', self.out.shape)
        rnn_out, (h,c) = self.rnn(out, (h,c))
        # print('rnnout1',rnn_out.shape, rnn_out.min(), rnn_out.max())
        rnn_out = self.readout(rnn_out) #added
        # print('rnnout2',rnn_out.shape, rnn_out.min(), rnn_out.max())
        stimulation = self.out_activation(rnn_out)
        # print('stimulation1',stimulation.shape, stimulation.min(), stimulation.max())
        if self.constrain:
            
            stimulation = stimulation*self.output_scaling
            # print('stimulation2',stimulation.shape, stimulation.min(), stimulation.max())
        # stimulation = self.out_activation(rnn_out)*self.output_scaling

        # print('stimulation', stimulation.shape)
        return stimulation, rnn_out, (h,c) #torch.Size([10, 2, 1000]),  torch.Size([10, 2, 1000]), (torch.Size([1, 2, 1000]), torch.Size([1, 2, 1000]))

#added
class E2E_Encoder_RNN_out3(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', hidden_size=512, num_layers=1, rnn_input_size=2000, constrained=[True,True,True]):
        super(E2E_Encoder_RNN_out3, self).__init__()
        self.output_scaling = out_scaling
        # self.out_activation = {'tanh': nn.Tanh(), ## NOTE: simulator expects only positive stimulation values 
        #                        'sigmoid': nn.Sigmoid(),
        #                        'relu': nn.ReLU(),
        #                        'softmax':nn.Softmax(dim=1)}[out_activation]

        self.output_scaling_amp = 500.0e-6 #out_scaling
        self.output_scaling_pw = 500.0e-6 #out_scaling
        self.output_scaling_freq = 500 #out_scaling

        self.constrained = constrained

        # Model
        # self.enc = ResNetEncoder()

        self.enc = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                #    *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(8,16,3,2,1,resample_out=None),
                                #    *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,2,1,resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                #    ResidualBlock(32, resample_out=None),
                                #    ResidualBlock(32, resample_out=None),
                                   *convlayer(32,out_channels,3,1,1)) #, #*convlayer(32,16,3,1,1),
                                #    nn.Conv2d(16,out_channels,3,1,1)) #, #nn.Conv2d(16,1,3,1,1),
                                #   nn.Flatten(start_dim=2), #nn.Flatten(),
                                #    nn.Linear(1024,n_electrodes),
                                #    self.out_activation)

        self.flatten = nn.Flatten(start_dim=2)
        self.dense = nn.Linear(1024,rnn_input_size)
        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=num_layers)

        self.readout_amplitude = nn.Linear(hidden_size, n_electrodes)
        self.readout_pulse_width = nn.Linear(hidden_size, n_electrodes)
        self.readout_frequency = nn.Linear(hidden_size, n_electrodes)

        if self.constrained[0]:
            self.activation_amp = nn.Sigmoid()
        else: 
            self.activation_amp = nn.ReLU()
        
        if self.constrained[1]:
            self.activation_pw = nn.Sigmoid()
        else: 
            self.activation_pw = nn.ReLU()

        if self.constrained[2]:
            self.activation_freq = nn.Sigmoid()
        else: 
            self.activation_freq = nn.ReLU()
        


    def forward(self, x, hidden_state):
        h = hidden_state[0]
        c = hidden_state[1]
        out = self.enc(x)
        # print('encoded', self.out.shape)
        out= self.flatten(out)
        # print('flattened', self.out.shape)
        out = self.dense(out)
        # print('dense', self.out.shape)
        out= out.permute(1,0,2)
        # print('permute', self.out.shape)
        rnn_out, (h,c) = self.rnn(out, (h,c))
        # print('rnnout', rnn_out.shape)

        amplitude = self.readout_amplitude(rnn_out) #added
        pulse_width = self.readout_pulse_width(rnn_out) #added
        frequency = self.readout_frequency(rnn_out) #added
        # print('amp', amplitude.shape)
        # print('pulse_width', pulse_width.shape)
        # print('frequency', frequency.shape)

        amplitude = self.activation_amp(amplitude)
        pulse_width = self.activation_pw(pulse_width)
        frequency = self.activation_freq(frequency)

        if self.constrained[0]:
            amplitude = amplitude*self.output_scaling_amp
        if self.constrained[1]:
            pulse_width = pulse_width*self.output_scaling_pw
        if self.constrained[2]:
            frequency = frequency*self.output_scaling_freq
        
        # print('stimulation', stimulation.shape)
        return amplitude, pulse_width, frequency, rnn_out, (h,c) #torch.Size([10, 2, 1000]),  torch.Size([10, 2, 1000]), (torch.Size([1, 2, 1000]), torch.Size([1, 2, 1000]))

#added
class E2E_Encoder_RNN_out32(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', hidden_size=512, num_layers=1, rnn_input_size=2000, constrained=[True,True,True], constrained_based_on='freq', constraint_coeff=2):
        super(E2E_Encoder_RNN_out32, self).__init__()

        self.constrained_based_on = constrained_based_on
        self.constraint_coeff = constraint_coeff
        # self.output_scaling = out_scaling
        self.output_scaling_amp_limit = 500.0e-6 #out_scaling
        self.output_scaling_pw_limit = 500.0e-6 #500.0e-6 #out_scaling
        self.output_scaling_freq_limit = 500 #out_scaling

        # self.output_scaling = out_scaling
        # self.out_activation = {'tanh': nn.Tanh(), ## NOTE: simulator expects only positive stimulation values 
        #                        'sigmoid': nn.Sigmoid(),
        #                        'relu': nn.ReLU(),
        #                        'softmax':nn.Softmax(dim=1)}[out_activation]

        # self.output_scaling_amp = 500.0e-6 #out_scaling
        # self.output_scaling_pw = 500.0e-6 #out_scaling
        # self.output_scaling_freq = 500 #out_scaling

        self.constrained = constrained

        # Model
        # self.enc = ResNetEncoder()

        self.enc = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                #    *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(8,16,3,2,1,resample_out=None),
                                #    *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,2,1,resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                #    ResidualBlock(32, resample_out=None),
                                #    ResidualBlock(32, resample_out=None),
                                   *convlayer(32,out_channels,3,1,1)) #, #*convlayer(32,16,3,1,1),
                                #    nn.Conv2d(16,out_channels,3,1,1)) #, #nn.Conv2d(16,1,3,1,1),
                                #   nn.Flatten(start_dim=2), #nn.Flatten(),
                                #    nn.Linear(1024,n_electrodes),
                                #    self.out_activation)

        self.flatten = nn.Flatten(start_dim=2)
        self.dense = nn.Linear(1024,rnn_input_size)
        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=num_layers)

        self.readout_amplitude = nn.Linear(hidden_size, n_electrodes)
        self.readout_pulse_width = nn.Linear(hidden_size, n_electrodes)
        self.readout_frequency = nn.Linear(hidden_size, n_electrodes)

        if self.constrained[0]:
            self.activation_amp = nn.Sigmoid()
        else: 
            self.activation_amp = nn.ReLU()
        
        if self.constrained[1]:
            self.activation_pw = nn.Sigmoid()
        else: 
            self.activation_pw = nn.ReLU()

        if self.constrained[2]:
            self.activation_freq = nn.Sigmoid()
        else: 
            self.activation_freq = nn.ReLU()
        


    def forward(self, x, hidden_state):
        h = hidden_state[0]
        c = hidden_state[1]
        out = self.enc(x)
        # print('encoded', self.out.shape)
        out= self.flatten(out)
        # print('flattened', self.out.shape)
        out = self.dense(out)
        # print('dense', self.out.shape)
        out= out.permute(1,0,2)
        # print('permute', self.out.shape)
        rnn_out, (h,c) = self.rnn(out, (h,c))

        amplitude = self.readout_amplitude(rnn_out) #added
        pulse_width = self.readout_pulse_width(rnn_out) #added
        frequency = self.readout_frequency(rnn_out) #added

        amplitude = self.activation_amp(amplitude)
        pulse_width = self.activation_pw(pulse_width)
        frequency = self.activation_freq(frequency)

        if self.constrained[0]:
            amplitude = amplitude*self.output_scaling_amp_limit
        
        if self.constrained_based_on == 'freq':
            #on fr
            if self.constrained[2]:
                frequency = frequency*self.output_scaling_freq_limit

            if self.constrained[1]:
                # output_scaling_pw = dclamp(1/frequency, 0, self.output_scaling_pw_limit)
                # pulse_width = pulse_width*output_scaling_pw
                output_scaling_pw = torch.clamp(1/(self.constraint_coeff*frequency), 0, self.output_scaling_pw_limit)
                pulse_width = pulse_width * output_scaling_pw.detach()

        elif self.constrained_based_on == 'pw':
            #on pw
            if self.constrained[1]:
                pulse_width = pulse_width*self.output_scaling_pw_limit

            if self.constrained[2]:
                # output_scaling_freq = dclamp(1/pulse_width, 0, self.output_scaling_freq_limit)
                # frequency = frequency*output_scaling_freq
                output_scaling_freq = torch.clamp(1/(self.constraint_coeff*pulse_width), 0, self.output_scaling_freq_limit)
                frequency = frequency * output_scaling_freq.detach()
        
        # print('stimulation', stimulation.shape)
        return amplitude, pulse_width, frequency, rnn_out, (h,c) #torch.Size([10, 2, 1000]),  torch.Size([10, 2, 1000]), (torch.Size([1, 2, 1000]), torch.Size([1, 2, 1000]))


class E2E_Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SVP representation
    out: (128x128) Reconstruction
    """
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(E2E_Decoder, self).__init__()

        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,16,3,1,1),
                                   *convlayer(16,32,3,1,1),
                                   *convlayer(32,64,3,2,1),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   *convlayer(64,32,3,1,1),
                                   nn.Conv2d(32,out_channels,3,1,1),
                                   self.out_activation)

    def forward(self, x):
        return self.model(x)

# class E2E_RealisticPhospheneSimulator(nn.Module):
#     """A realistic simulator, using stimulation vectors to form a phosphene representation
#     in: a 1024 length stimulation vector
#     out: 256x256 phosphene representation
#     """
#     def __init__(self, cfg, params, r, phi):
#         super(E2E_RealisticPhospheneSimulator, self).__init__()
#         self.simulator = GaussianSimulator(params, r, phi, batch_size=cfg.batch_size, device=cfg.device)
        
#     def forward(self, stimulation):
#         phosphenes = self.simulator(stim_amp=stimulation).clamp(0,1)
#         phosphenes = phosphenes.view(phosphenes.shape[0], 1, phosphenes.shape[1], phosphenes.shape[2])
#         return phosphenes

class ZhaoEncoder(nn.Module):
    # def __init__(self, in_channels=3,n_electrodes=638, out_channels=1):
    #added
    def __init__(self, in_channels=3,n_electrodes=638, out_channels=1, out_scaling=128e-6, constrained=True):
        super(ZhaoEncoder, self).__init__()

        self.output_scaling = out_scaling

        self.constrain = constrained

        self.model = nn.Sequential(
            # *convlayer3d(in_channels,32,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(in_channels,32,3,(1,2,2),1, resample_out=None), #to get rid of maxpool
            # *convlayer3d(32,48,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(32,48,3,(1,2,2),1, resample_out=None), #to get rid of maxpool
            *convlayer3d(48,64,3,1,1),
            *convlayer3d(64,1,3,1,1),

            nn.Flatten(start_dim=3),
            nn.Linear(1024,n_electrodes))
        
        if self.constrain:
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.ReLU()
            #changed for 0-128 microamper limit, original was different
            
            #Ssuboptimal learning changed again
            # nn.Tanh()
        # )

    def forward(self, x):
        out = self.model(x)
        out = self.out_activation(out) #added
        out = out.squeeze(dim=1)
        # self.out = self.out*1e-4
        if self.constrain:
            out = out*self.output_scaling
        #added for tanh
        # self.out = 0.5*(self.out+torch.ones(self.out.shape).to(self.out.device))
        return out

#added
class ZhaoEncoder_clamp(nn.Module):
    # def __init__(self, in_channels=3,n_electrodes=638, out_channels=1):
    #added
    def __init__(self, in_channels=3,n_electrodes=638, out_channels=1, out_scaling=128e-6):
        super(ZhaoEncoder_clamp, self).__init__()

        self.output_scaling = out_scaling

        self.model = nn.Sequential(
            # *convlayer3d(in_channels,32,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(in_channels,32,3,(1,2,2),1, resample_out=None), #to get rid of maxpool
            # *convlayer3d(32,48,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(32,48,3,(1,2,2),1, resample_out=None), #to get rid of maxpool
            *convlayer3d(48,64,3,1,1),
            *convlayer3d(64,1,3,1,1),

            nn.Flatten(start_dim=3),
            nn.Linear(1024,n_electrodes),
            # nn.ReLU()
            #changed for 0-128 microamper limit, original was different

            
            # nn.Sigmoid() #COMMENTED changed for clamp
            
            #Ssuboptimal learning changed again
            # nn.Tanh()
        )
        # self.clamp= DifferentiableClamp()
        self.min_amp = 0 
        self.max_amp = out_scaling

    def forward(self, x):
        out = self.model(x)
        # print('bef clamp', out.min(), out.max())
        out = dclamp(out, self.min_amp, self.max_amp) #added
        # print('aft clamp', out.min(), out.max())
        out = out.squeeze(dim=1)
        # self.out = self.out*1e-4
        # out = out*self.output_scaling
        #added for tanh
        # self.out = 0.5*(self.out+torch.ones(self.out.shape).to(self.out.device))
        return out

class ZhaoEncoder2(nn.Module):
    # def __init__(self, in_channels=3,n_electrodes=638, out_channels=1):
    #added/changed
    def __init__(self, in_channels=3,n_electrodes=638, out_channels=1, sequence_length=1, output_scaling=1e-4):
        super(ZhaoEncoder2, self).__init__()

        self.model = nn.Sequential(
            *convlayer3d(in_channels,32,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(32,48,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(48,64,3,1,1),
            *convlayer3d(64,1,3,1,1),

            nn.Flatten(start_dim=2), #nn.Flatten(start_dim=3),
            nn.Linear(1024 * sequence_length ,n_electrodes),#nn.Linear(1024,n_electrodes),
            nn.ReLU(inplace=False) #nn.ReLU()
        )

    def forward(self, x):
        # for i in range(len(self.model)):
        # print(i, self.model[i])
        self.out = self.model[0](x)
        self.out = self.model[1](self.out)
        self.out = self.model[2](self.out)
        self.out = self.model[3](self.out)
        self.out = self.model[4](self.out)
        self.out = self.model[5](self.out)
        self.out = self.model[6](self.out)
        self.out = self.model[7](self.out)
        self.out = self.model[8](self.out)
        self.out = self.model[9](self.out)
        self.out = self.model[10](self.out)
        self.out = self.model[11](self.out)
        self.out = self.model[12](self.out)
        self.out = self.model[13](self.out)
        self.out = self.model[14](self.out)
        self.out = self.model[15](self.out)
        self.out = self.model[16](self.out)
        self.out = self.out.squeeze(dim=1)
        self.out = self.out*1e-4 
        return self.out

class ZhaoEncoder_out3(nn.Module):
    # def __init__(self, in_channels=3,n_electrodes=638, out_channels=1):
    #added
    def __init__(self, in_channels=3,n_electrodes=638, out_channels=3, out_scaling=128e-6, constrained=[True, True, True]):
        super(ZhaoEncoder_out3, self).__init__()

        # self.output_scaling = out_scaling
        self.output_scaling_amp = 500.0e-6 #out_scaling
        self.output_scaling_pw = 500.0e-6 #out_scaling
        self.output_scaling_freq = 500 #out_scaling

        self.constrained = constrained
        print('self.constrained', self.constrained)

        self.model = nn.Sequential(
            # *convlayer3d(in_channels,32,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(in_channels,32,3,(1,2,2),1, resample_out=None), #to get rid of maxpool
            # *convlayer3d(32,48,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(32,48,3,(1,2,2),1, resample_out=None), #to get rid of maxpool
            *convlayer3d(48,64,3,1,1),
            *convlayer3d(64,1,3,1,1),
            nn.Flatten(start_dim=3)
            ) #,
        
        # self.linear =   nn.Linear(1024,n_electrodes) #,
        self.amplitude =   nn.Linear(1024,n_electrodes)
        self.pulse_width =   nn.Linear(1024,n_electrodes)
        self.frequency =   nn.Linear(1024,n_electrodes)
            # nn.ReLU()
            #changed for 0-128 microamper limit, original was different
            
        if self.constrained[0]:
            self.activation_amp = nn.Sigmoid()
        else: 
            self.activation_amp = nn.ReLU()
        
        if self.constrained[1]:
            self.activation_pw = nn.Sigmoid()
        else: 
            self.activation_pw = nn.ReLU()

        if self.constrained[2]:
            self.activation_freq = nn.Sigmoid()
        else: 
            self.activation_freq = nn.ReLU()

        # self.activation_amp = nn.Sigmoid()
        # self.activation_pw = nn.ReLU()
        # self.activation_freq = nn.ReLU()
            
            #Ssuboptimal learning changed again
            # nn.Tanh()
        # )

    def forward(self, x):
        out = self.model(x)
        # print('model',out.shape)
        amplitude = self.amplitude(out)
        pulse_width = self.pulse_width(out)
        frequency = self.frequency(out)
        # print('amplitude', amplitude.shape)
        # print('pulse_width', pulse_width.shape)
        # print('frequency', frequency.shape)

        # amplitude = self.activation(amplitude)
        # pulse_width = self.activation(pulse_width)
        # frequency = self.activation(frequency)
        amplitude = self.activation_amp(amplitude)
        pulse_width = self.activation_pw(pulse_width)
        frequency = self.activation_freq(frequency)

        # out = out.squeeze(dim=1)
        amplitude = amplitude.squeeze(dim=1)
        pulse_width = pulse_width.squeeze(dim=1)
        frequency = frequency.squeeze(dim=1)
        # print('sq amp',amplitude.shape)
        # print('sq pw',pulse_width.shape)
        # print('sq freq',frequency.shape)
        # self.out = self.out*1e-4
        # out = out*self.output_scaling
        if self.constrained[0]:
            amplitude = amplitude*self.output_scaling_amp
        if self.constrained[1]:
            pulse_width = pulse_width*self.output_scaling_pw
        if self.constrained[2]:
            frequency = frequency*self.output_scaling_freq
        #added for tanh
        # self.out = 0.5*(self.out+torch.ones(self.out.shape).to(self.out.device))
        return amplitude, pulse_width, frequency
        # return (out, out, out)

#added
class ZhaoEncoder_out32(nn.Module):
    # def __init__(self, in_channels=3,n_electrodes=638, out_channels=1):
    #added
    def __init__(self, in_channels=3,n_electrodes=638, out_channels=3, out_scaling=128e-6, constrained=[True, True, True], constrained_based_on='freq', constraint_coeff=2):
        super(ZhaoEncoder_out32, self).__init__()

        self.constrained_based_on = constrained_based_on
        self.constraint_coeff = constraint_coeff

        self.output_scaling_amp_limit = 500.0e-6 #out_scaling
        self.output_scaling_pw_limit = 500.0e-6 #500.0e-6 #out_scaling
        self.output_scaling_freq_limit = 500 #out_scaling


        self.constrained = constrained
        # print('self.constrained', self.constrained)

        self.model = nn.Sequential(
            # *convlayer3d(in_channels,32,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(in_channels,32,3,(1,2,2),1, resample_out=None), #to get rid of maxpool
            # *convlayer3d(32,48,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(32,48,3,(1,2,2),1, resample_out=None), #to get rid of maxpool
            *convlayer3d(48,64,3,1,1),
            *convlayer3d(64,1,3,1,1),
            nn.Flatten(start_dim=3)
            ) #,
        
        # self.linear =   nn.Linear(1024,n_electrodes) #,
        self.amplitude =   nn.Linear(1024,n_electrodes)
        self.pulse_width =   nn.Linear(1024,n_electrodes)
        self.frequency =   nn.Linear(1024,n_electrodes)
            # nn.ReLU()
            #changed for 0-128 microamper limit, original was different
            
        if self.constrained[0]:
            self.activation_amp = nn.Sigmoid()
        else: 
            self.activation_amp = nn.ReLU()
        
        if self.constrained[1]:
            self.activation_pw = nn.Sigmoid()
        else: 
            self.activation_pw = nn.ReLU()

        if self.constrained[2]:
            self.activation_freq = nn.Sigmoid()
        else: 
            self.activation_freq = nn.ReLU()

        # self.activation_amp = nn.Sigmoid()
        # self.activation_pw = nn.ReLU()
        # self.activation_freq = nn.ReLU()
            
            #Ssuboptimal learning changed again
            # nn.Tanh()
        # )

    def forward(self, x):
        out = self.model(x)
        # print('model',out.shape)
        amplitude = self.amplitude(out)
        pulse_width = self.pulse_width(out)
        frequency = self.frequency(out)
        # print('amplitude', amplitude.shape)
        # print('pulse_width', pulse_width.shape)
        # print('frequency', frequency.shape)

        # amplitude = self.activation(amplitude)
        # pulse_width = self.activation(pulse_width)
        # frequency = self.activation(frequency)
        amplitude = self.activation_amp(amplitude)
        pulse_width = self.activation_pw(pulse_width)
        frequency = self.activation_freq(frequency)

        # out = out.squeeze(dim=1)
        amplitude = amplitude.squeeze(dim=1)
        pulse_width = pulse_width.squeeze(dim=1)
        frequency = frequency.squeeze(dim=1)
        # print('sq amp',amplitude.shape)
        # print('sq pw',pulse_width.shape)
        # print('sq freq',frequency.shape)
        # self.out = self.out*1e-4
        # out = out*self.output_scaling
        if self.constrained[0]:
            amplitude = amplitude*self.output_scaling_amp_limit
        
        if self.constrained_based_on == 'freq':
            #on fr
            if self.constrained[2]:
                frequency = frequency*self.output_scaling_freq_limit

            if self.constrained[1]:
                # output_scaling_pw = dclamp(1/frequency, 0, self.output_scaling_pw_limit)
                output_scaling_pw = torch.clamp(1/(self.constraint_coeff*frequency), 0, self.output_scaling_pw_limit)
                pulse_width = pulse_width * output_scaling_pw.detach()

        elif self.constrained_based_on == 'pw':
            #on pw
            if self.constrained[1]:
                # print('limit',self.output_scaling_pw_limit)
                # print('pulse_width bef',  pulse_width.min(),pulse_width.max(), pulse_width.isnan().any())
                pulse_width = pulse_width*self.output_scaling_pw_limit
                # print('pulse_width aft', pulse_width.min(),pulse_width.max(), pulse_width.isnan().any())

            if self.constrained[2]:
                # output_scaling_freq = dclamp(1/pulse_width, 0, self.output_scaling_freq_limit)
                output_scaling_freq = torch.clamp(1/(self.constraint_coeff*pulse_width), 0, self.output_scaling_freq_limit)
                frequency = frequency * output_scaling_freq.detach()

        #added for tanh
        # self.out = 0.5*(self.out+torch.ones(self.out.shape).to(self.out.device))
        return amplitude, pulse_width, frequency
        # return (out, out, out)

class ZhaoDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(ZhaoDecoder, self).__init__()
        
        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        self.model = nn.Sequential(
            *convlayer3d(in_channels,16,3,1,1),
            *convlayer3d(16,32,3,1,1),
            *convlayer3d(32,64,3,(1,2,2),1),
            *convlayer3d(64,32,3,1,1),
            nn.Conv3d(32,out_channels,3,1,1),
            self.out_activation
        )

    def forward(self, x):
        out = self.model(x)
        return out

#added

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        # with self.name_scope():
        self.layers = {}
        self.layers.items()
        self.channel_list = [24,32,64,128]
        self.input_list = [4,24,32,64]
        for i, channels in enumerate(self.channel_list):
            layer = str(i)
            self.layers['conv'+layer] = nn.Conv2d(in_channels=self.input_list[i],out_channels=channels, kernel_size=3, stride=1, padding=1)
            self.layers['max'+layer] = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
            self.layers['res'+layer+'_0'] =  ResidualBlock2(channels)
            self.layers['res'+layer+'_1'] =  ResidualBlock2(channels)

        # for key, val in self.layers.items():
        #     self.register_child(self.layers[key])

    def forward(self, x):
        for i, channels in enumerate(self.channel_list):
            layer = str(i)
            x = F.relu(x)
            x = self.layers['conv'+layer](x)
            x = self.layers['max'+layer](x)
            x = self.layers['res'+layer+'_0'](x)
            x = self.layers['res'+layer+'_1'](x)
        return x


class ResidualBlock2(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock2, self).__init__()
        # with self.name_scope():
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

        self.in_channels = in_channels

    def forward(self, x):
        h = F.relu(x)
        h = self.conv0(h)
        h = F.relu(h)
        h = self.conv1(h)
        x = h+x
        return x


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)