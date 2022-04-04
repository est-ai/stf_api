import os
import errno
import torch
import gc
import sys
from .util import *
from .s2f_dir.src import autoencoder as ae 


g_fix_seed = False


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


def __init_fix_seed(random_seed, verbose=False):
    global g_fix_seed
    if g_fix_seed == True:
        return
    
    if verbose:
        print('fix seed')
    fix_seed(random_seed)
    g_fix_seed = True

        
def create_model(config_path, checkpoint_path, work_root_path, device, verbose=False):
    __init_fix_seed(random_seed=1234, verbose=verbose)
    if verbose:
        print('load model')

    if not os.path.exists(config_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_path)
        
    args = read_config(config_path)
    model = ae.Speech2Face(3, (3, args.img_size, args.img_size), (1, 96, args.mel_step_size))
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        gpus = list(range(torch.cuda.device_count()))
        print('Multi GPU activate, gpus : ', gpus)
        model = torch.nn.DataParallel(model, device_ids=gpus)
        model.to(device)
        model.eval()
    else:
        model.to(device).eval()
    
    model_data = ModelInfo(model=model, args=args, device=device,
                           work_root_path=work_root_path,
                           config_path=config_path,
                           checkpoint_path=checkpoint_path,
                           verbose=verbose)
    del checkpoint
    gc.collect()
    if verbose:
        print('load model complete')
    
    return model_data


