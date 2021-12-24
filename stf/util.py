import json
from addict import Dict
from pathlib import Path


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


