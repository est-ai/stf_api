import os
import errno
import torch
import gc
import sys
from .util import *
from .s2f_dir.src import autoencoder as ae 


class ModelInfo():
    def __init__(self, model, args, device, work_root_path, config_path, checkpoint_path, verbose=False):
        self.model = model
        self.args = args
        self.device = device 
        # snow : 아래는 debuging 을 위해 저장해 두는 것
        self.work_root_path = work_root_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        
    def __del__(self):
        if self.verbose:
            print('del model , gc:',  sys.getrefcount(self.model))
        del self.model
        
        
def create_model(config_path, checkpoint_path, work_root_path, device, verbose=False):
    if verbose:
        print('load model')

    if not os.path.exists(config_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_path)
        
    args = read_config(config_path)
    model = ae.Speech2Face(3, (3, args.img_size, args.img_size), (1, 96, args.mel_step_size))
    #print('1 model , gc:',  sys.getrefcount(model), ', referres cnt:', len(gc.get_referrers(model)))
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    #print('2 model , gc:',  sys.getrefcount(model), ', referres cnt:', len(gc.get_referrers(model)))
    
    model_data = ModelInfo(model=model, args=args, device=device,
                           work_root_path=work_root_path,
                           config_path=config_path,
                           checkpoint_path=checkpoint_path,
                           verbose=verbose)
    #print('3 model , gc:',  sys.getrefcount(model), ', referres cnt:', len(gc.get_referrers(model)))
    del checkpoint
    gc.collect()
    if verbose:
        print('load model complete')
    
    #print('4 model , gc:',  sys.getrefcount(model), ', referres cnt:', len(gc.get_referrers(model)))
    return model_data


