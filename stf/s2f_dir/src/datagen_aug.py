import os, random
from glob import glob, escape
from pathlib import Path
import numpy as np

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import pdb

from .mask_history import calc_poly
from .transform_history import mask_img_trsfs 

# snow : LipGanDS.__init__ 에서 계하도록 변경됨
#half_window_size = 4
# parameter로 받도록 수정
#mel_step_size = 27


def frame_id(fname):
    return int(os.path.basename(fname).split('_')[0])

def choose_ip_frame(frames, gt_frame, num_ips, phase):
    # 생성하려는 프레임과 동일한 프레임을 친구프레임으로 준다.
    if phase == 'val':
        d = os.path.dirname(frames[0])
        return [os.path.join(d, gt_frame) for _ in range(num_ips)]
    # 0번프레임을 무조건 친구 프레임으로 준다.
    #if phase == 'val':
    #    return [frames[0] for _ in range(num_ips)]

    # 랜덤하게 골라준다.
    frames = [f for f in frames if np.abs(frame_id(gt_frame) - frame_id(f)) >= 12]
    if len(frames) < num_ips:
        return None
    random.shuffle(frames)
    return frames[:num_ips]

def get_audio_segment(center_frame, spec, mel_step_size, mel_ps, fps, half_window_size):
    center_frame_id = frame_id(center_frame)
    start_frame_id = center_frame_id - half_window_size

    #start_idx = int((80./25.) * start_frame_id) # 25 is fps of LRS2
    start_idx = int((float(mel_ps)/float(fps)) * start_frame_id) # mel, frame per sec 에 따라 계산
    if start_idx < 0:
        spec = np.pad(spec, ((0,0), (-start_idx, 0)), mode='edge') 
        start_idx = 0

    end_idx = start_idx + mel_step_size
    if spec.shape[1] < end_idx:
        spec = np.pad(spec, ((0,0), (0, end_idx-spec.shape[1])), mode='edge') 
    
    #print('center_frame_id:', center_frame_id, ', mel [s,e]', start_idx, end_idx, ', mel shape:', spec.shape)
       
    return spec[:, start_idx : end_idx]

def inter_alg(target_size, img):
    if isinstance(target_size, tuple):
        w, h = target_size
    else:
        w, h = target_size, target_size
    return inter_alg_(w,h, img)
        
def inter_alg_(w, h, img):
    if w*h < img.shape[0] * img.shape[1]:
        return cv2.INTER_AREA
    else:
        return cv2.INTER_CUBIC
    
def resize_adapt(args, img):
    sz = args.img_size
    board = np.full((sz, sz, 3), 128, np.uint8)
    h, w = img.shape[:2]
    if True:
    #if sz < max(h, w):
        r = sz/max(h,w)
        h, w = int(round(r*h)), int(round(r*w))
        img = cv2.resize(img, (w, h), inter_alg(sz, img))
    board[(sz-h)//2:(sz-h)//2+h, (sz-w)//2:(sz-w)//2+w] = img
    return board

def masking(im, pts):
    h, w = im.shape[:2]
    im = cv2.fillPoly(im, [pts], (128,128,128))
    return im

def id_map(x, rng = None):
    return x

g_cached_fps = {}
g_cached_frames = {}
g_cached_mels = {}
g_cached_pickle = {}

class LipGanDS(Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.mel_step_size = args.mel_step_size
        self.mel_ps = args.mel_ps
        self.images = args.train_images if phase =='train' else args.val_images[-len(args.val_images)//2:]
        self.val_audios = args.val_images[:-len(args.val_images)//2]
        self.mask_ver = list(args.mask_ver) if isinstance(args.mask_ver, (list, tuple)) else [args.mask_ver]
        self.num_ips = args.num_ips
        self.mel_trsf_ver = args.mel_trsf_ver
        self.mel_norm_ver = args.mel_norm_ver
        
        #self.half_window_size = self.calc_half_window_size(args.fps) 
        print(f'mel_step_size:{self.mel_step_size}, mel_ps:{args.mel_ps}')
        #print(f'half_window_size:{self.half_window_size}, fps:{args.fps}, a_frame_secs:{1/args.fps}')
        
        if phase == 'val' or args.mask_img_trsf_ver < 0:
            self.mask_img_trsf = id_map
        else:
            self.mask_img_trsf = mask_img_trsfs[args.mask_img_trsf_ver]
            
            
    
    def calc_half_window_size(self, fps):
        mel_step_secs = self.mel_step_size * 1.0/self.mel_ps
        a_frame_secs = 1.0/fps
        return int(mel_step_secs / a_frame_secs / 2.0)
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        while(1):
            
            ref_image_only = self.phase =='val'
            ret_0 = self.choose(idx, self.images, ref_image_only)
            if self.phase == 'train':
                if ret_0 is not None:
                    return ret_0
            else:
                if ret_0 is not None:
                    ret_1 = self.choose(idx, self.val_audios, gt_audio_only=True)
                    if ret_1 is not None:
                        img_gt_0, mel_0, ips_0 = ret_0
                        img_gt_1, mel_1, ips_1 = ret_1
                        return img_gt_1, mel_1, ips_0
            
            if self.phase == 'train':
                idx += 1023
            else:
                idx += 1
            idx%=len(self.images)
            
    def read_fps(self, dir_name):
        if str(dir_name) not in g_cached_fps:
            with open(dir_name/'fps.txt') as f:
                fps = float(f.read())
            return fps
            g_cached_fps[str(dir_name)] = fps
        return g_cached_fps[str(dir_name)]
    
    def get_frames(self, dir_name):
        if str(dir_name) not in g_cached_frames:
            frames = glob(escape(str(dir_name)) + '/*.jpg')
            return frames
            g_cached_frames[str(dir_name)] = frames
        return g_cached_frames[str(dir_name)]
    
    def load_mel(self, dir_name):
        if str(dir_name)  not in g_cached_mels:
            mel_fname = dir_name / 'mels.npz'
            if 0 < self.mel_norm_ver:
                mel_fname = dir_name / f'mels_v{self.mel_norm_ver}.npz'

            if self.mel_trsf_ver == 0:
                if random.randint(0,1) == 0:
                    mel_fnames = [dir_name/ f'mels_{i:02d}.npz' for i in range(10)]
                    mel_fname_2 = random.choice(mel_fnames)
                    if Path(mel_fname_2).exists():
                        mel_fname = mel_fname_2
                    
            with np.load(str(mel_fname)) as f:
                mel = f['spec']
            #mel = np.load(str(mel_fname))['spec']
            return mel
            g_cached_mels[str(dir_name)] = mel
        return g_cached_mels[str(dir_name)]
        
    def read_pickle(self, dir_name):
        if str(dir_name) not in g_cached_pickle:
            #print('pikcle_dir:', dir_name)
            df = pd.read_pickle(dir_name/'df_fan.pickle')
            preds = df.set_index('frame_idx')['cropped_pts2d']
            #g_cached_pickle[str(dir_name)] = preds
            return preds
        return g_cached_pickle[str(dir_name)]
    
    def choose(self, idx, images, ref_image_only=False, gt_audio_only=False):
        args = self.args
        img_name = Path(images[idx])
        
        gt_fname = img_name.name
        dir_name = img_name.parent
        
        sidx = frame_id(gt_fname)
        
        
        frames = self.get_frames(dir_name)
        if len(frames) < 12:
            return None
        
        if 'overfitting' in str(img_name):
            if sidx < 30 or len(frames) < (sidx + 30):
                return None
        
        if ref_image_only:
            mel = None
        else:
            mel = self.load_mel(dir_name)
            fps = self.read_fps(dir_name)
            mel = get_audio_segment(gt_fname, mel, self.mel_step_size, self.mel_ps,
                                    fps, self.calc_half_window_size(fps))
            
            if mel is None or mel.shape[1] != self.mel_step_size:
                return None
            
            if sum(np.isnan(mel.flatten())) > 0:
                return None
        
        ip_fnames = choose_ip_frame(frames, gt_fname, self.num_ips, self.phase)
        if ip_fnames is None:
            print('return None')
            return None
        
        rng = random.randint(0, 65536*65536)
        
        img_gt = cv2.imread(str(img_name))
        masked = self.mask_img_trsf(img_gt.copy(), rng)
        
        img_gt = resize_adapt(args, img_gt)
        img_gt = self.mask_img_trsf(img_gt, rng)
        
        
        if gt_audio_only:
            masked = np.zeros((16,16,3), np.uint8)
            pts = None
        else:
            preds = self.read_pickle(dir_name)
            if preds[sidx] is None:
                print(f'preds[{sidx}] is None:', dir_name)
                return None
            
            mask_ver = random.choice(self.mask_ver)
            randomness = False if self.phase == 'val' else True
            pts = calc_poly[mask_ver](preds[sidx], masked.shape[0], randomness)
            
            masked = masking(masked, pts)
        masked = resize_adapt(args, masked)
        
        img_ips = []
        for ip_fname in ip_fnames:
            #img_ip = cv2.imread(os.path.join(dir_name, ip_fname))
            img_ip = cv2.imread(ip_fname)
            img_ip = resize_adapt(args, img_ip)
            img_ip = self.mask_img_trsf(img_ip, rng)
            img_ip = img_ip * 2.0 / 255.0 - 1.0
            img_ips.append(img_ip)
            
        ips = np.concatenate([masked * 2.0 /255.0 - 1.0] + img_ips, axis=2)
        if mel is not None:
            mel = mel.astype(np.float32) 
        return (img_gt * 2.0 /255.0 - 1.0).astype(np.float32), mel, ips.astype(np.float32)


def datagen(args, phase, shuffle, drop_last = True):
    
    ds = LipGanDS(args, phase)
    batch_size = args.batch_size * args.ngpu
    print('batch_size_per_gpu:', args.batch_size)
    print('ngpu:', args.ngpu)
    print('batch_size_total:', batch_size)
    dl = DataLoader(dataset=ds, batch_size=batch_size, num_workers=args.num_workers*args.ngpu, shuffle=shuffle, drop_last=drop_last)
            
    def inner():
        
        while True:
            
            for img_gt_batch, mel_batch, ips_batch in dl:
    
                #img_gt_batch = np.asarray(img_gt_batch)
                img_gt_batch = img_gt_batch.numpy()
                #mel_batch    = np.expand_dims(np.asarray(mel_batch), 3)
                mel_batch    = np.expand_dims(mel_batch.numpy(), 3)
                #img_ip_batch = np.asarray(img_gt_masked_batch)
                #img_ip_batch = img_gt_masked_batch.numpy()
                #img_ip_batch = np.concatenate(ips + [img_ip_batch], axis=3)
                ips_batch = ips_batch.numpy()
    
                    #model = Model(inputs=[input_face, input_audio], outputs=prediction)
                #yield [2*(img_ip_batch/255.0 - 0.5), mel_batch], 2*(img_gt_batch/255.0 - 0.5)#, pts_batch
                yield [ips_batch, mel_batch], img_gt_batch #, pts_batch
                
    return inner, len(dl) # len(ds)//batch_size
        
