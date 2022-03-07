import json
from addict import Dict
from pathlib import Path
import pdb
import torch
import numpy as np
import random


def read_config(config_path):
    try:
        with open(config_path) as fd:
            conf = json.load(fd)
            conf = Dict(conf)
    except Exception as e:
        print('read config exception in ', config_path)
        raise e
    return conf


def get_preprocess_dir(work_root_path, name):
    return str(Path(work_root_path) / 'preprocess' / name)
    
    
def get_crop_mp4_dir(preprocess_dir, video_path):
    return f'{preprocess_dir}/crop_video_{Path(video_path).stem}'


class _CallBack(object):
    def __init__(self, callback, min_per, max_per, desc, verbose=False):
        assert(max_per > min_per)
        self.callback = callback
        self.min_per = min_per
        self.max_per = max_per
        if isinstance(callback, _CallBack):
            self.desc = callback.desc + '/' + desc
        else:
            self.desc = desc
        self.last_per = -1
        self.verbose = verbose
        self.callback_interval = 1
        
    def __call__(self, per):
        if self.callback is None: return
        my_per = self.min_per + (per+1) / 100.0 * (self.max_per - self.min_per)
        my_per = int(my_per)
        if my_per - self.last_per >= self.callback_interval:
            #if self.verbose:
            #    print(self.desc, ' : ', my_per)
            self.callback(my_per)
            self.last_per = my_per
        
def callback_inter(callback, min_per=0, max_per=100, desc='', verbose=False):
    assert(min_per >=0 and max_per >=0 and max_per > min_per)
    return _CallBack(callback, min_per, max_per, desc, verbose=verbose)


def callback_test():
    def callback(per):
        print('real callback', per)
        
    callback1 = callback_inter(callback, min_per=0, max_per=50, desc='1')
    callback2 = callback_inter(callback, min_per=50, max_per=90, desc='2')
    callback3 = callback_inter(callback, min_per=90, max_per=100, desc='3')
    #for i in range(0,101,10):
    #    callback1(i)
        
    callback11 = callback_inter(callback1, min_per=0, max_per=20, desc='a')
    callback12 = callback_inter(callback1, min_per=20, max_per=80, desc='b')
    callback13 = callback_inter(callback1, min_per=80, max_per=100, desc='c')
    
    for i in range(0,101,1):
        callback11(i)
    for i in range(0,101,1):
        callback12(i)
    for i in range(0,101,1):
        callback13(i)
        
    for i in range(0,101,1):
        callback2(i)
    for i in range(0,101,1):
        callback3(i)
        
        
def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

