
import math
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import cv2
import imageio
from scipy import stats
from tqdm.auto import tqdm
from moviepy.editor import AudioFileClip, ImageSequenceClip

from . import face_finder as ff
import face_alignment
import imageio_ffmpeg


g_detector_fan = None
g_detector_fan3d = None


def init_fan(device='cuda:0'):
    global g_detector_fan
    global g_detector_fan3d
    if g_detector_fan is None:
        g_detector_fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    if g_detector_fan3d is None:
        g_detector_fan3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)


def fan_box(pred, img, type3d):
    if type3d:
        xlist, ylist, _ = zip(*pred)
    else:
        xlist, ylist = zip(*pred)
    xlist = [int(round(x)) for x in xlist]
    ylist = [int(round(x)) for x in ylist]
    y1, y2, x1, x2 = [min(ylist), max(ylist), min(xlist), max(xlist)]
    size = max(y2 - y1 + 1, x2- x1 + 1)
    size = int(round(size))
    cx, cy = (x1+x2)/2, (y1+y2)/2
    x1, y1 = int(round(cx - size/2)), int(round(cy-size/2))
    x2, y2 = x1 + size -1, y1 + size -1
    
    y1 = max(0, y1)
    y2 = min(img.shape[0], y2)
    x1 = max(0, x1)
    x2 = min(img.shape[1], x2)
    return (x1, y1, x2, y2)
    
def face_detect_fan_(img, type3d):
    global g_detector_fan
    global g_detector_fan3d

    # snow : init_fan 을 미리 불러주지 않았으면 여기서 불리도록한다.
    init_fan()
    
    if type3d:
        preds = g_detector_fan3d.get_landmarks(img)
    else:
        preds = g_detector_fan.get_landmarks(img)
    
    preds = [(fan_box(p, img, type3d), p) for p in preds]
    preds = [((b[2]-b[0])*(b[3]-b[1]), b, p) for b, p in preds]
    preds = sorted(preds)
    area, (x1, y1, x2, y2), pred = preds[-1]
    return np.round((pred)).astype(np.int), np.array([x1, y1, x2, y2]) 


def face_detect_fan(img, type3d = False):
    try:
        return face_detect_fan_(img, type3d)
    except:
        return None, None 
    
def get_anchor_box(df_anchor, offset_y, margin, size_stride = 32, verbose=False):
    # 면적 평균을 구하고 너무(?) 작거나 큰 얼굴은 제거
    desc = df_anchor['area'].describe()
    area_25, area_75 = desc['25%'], desc['75%']
    df_anchor = df_anchor.query('@area_25 < area and area < @area_75')
    
    # z score로 아웃라이어 제거하고 평균 박스 구하기
    boxes = np.array(df_anchor['box'].values.tolist())
    center_xs = boxes[:,[0,2]].mean(axis=1)
    center_ys = boxes[:,[1,3]].mean(axis=1)
    size_xs   = boxes[:,2] - boxes[:,0]
    size_ys   = boxes[:,3] - boxes[:,1]
    center_x = np.mean([x for z, x in zip(stats.zscore(center_xs), center_xs) if abs(z) < 3]).round().astype(np.int)
    center_y = np.mean([y for z, y in zip(stats.zscore(center_ys), center_ys) if abs(z) < 3])
    center_y = int(round(center_y*(1+offset_y)))
    size_x   = np.mean([x for z, x in zip(stats.zscore(size_xs),   size_xs) if abs(z) < 3]).round().astype(np.int)
    size_y   = np.mean([y for z, y in zip(stats.zscore(size_ys),   size_ys) if abs(z) < 3]).round().astype(np.int)
    SS = size_stride
    size_step_x = int(math.ceil((size_x * (1+margin))/SS)*SS)
    size_step_y = int(math.ceil((size_y * (1+margin))/SS)*SS)
    
    x1 = center_x - int(size_step_x * 0.5)
    y1 = center_y - int(size_step_y * 0.5)
    
    y1 = max(0, y1)
    
    mean_box = [x1, y1, x1+size_step_x-1, y1+size_step_y-1]
    if verbose:
        print('mean_box:', mean_box, '  width:', size_step_x, ' height:', size_step_y)
    return mean_box

def df_fan_info(frames, box, verbose=False):
    x1, y1, x2, y2 = box
    
    def fan_info(f):
        face = f[y1:y2+2, x1:x2+1]
        pts2d, box = face_detect_fan(face)
        #pts3d, _   = face_detect_fan(face, type3d=True)
        pts3d = None
        return box, pts2d, pts3d
    
    def to_full(box, pts2d, pts3d, x1y1):
        if box is not None:
            box   = (box.reshape(-1,2) + x1y1).reshape(-1)
        if pts2d is not None:
            pts2d = pts2d + x1y1
        if pts3d is not None:
            pts3d = pts3d + (x1y1 +(0,))
        return box, pts2d, pts3d
    
    fi = [fan_info(frames[idx]) for idx in tqdm(frames, desc='■ fan ', disable=not verbose)]
    fi = [to_full(*info, (x1, y1)) for info in fi]
    
    df = pd.DataFrame(fi, columns=['box', 'pts2d', 'pts3d'])
    df['frame_idx'] = list(frames.keys())
    return df

def crop(frames, df_fan, offset_y, margin):
    df_fan = df_fan.copy()
    
    #ToDo: None을 제거해야 됨. crash 발생
    pts2ds = [e for e in df_fan['pts2d'].values if e is not None]
    if len(pts2ds):
        pts2ds = np.stack(pts2ds)
        x1, y1 = pts2ds[:,:,0].min(), pts2ds[:,:,1].min()
        x2, y2 = pts2ds[:,:,0].max(), pts2ds[:,:,1].max()
    else:
        return None, None
    
    cx, cy = (x1+x2)/2, (y1+y2)/2
    sx, sy = (x2-x1+1)*(1+margin), (y2-y1+1)*(1+margin)
    x1, y1 = cx-sx/2, cy-sy/2
    x2, y2 = cx+sx/2, cy+sy/2
    
    
    size = (x2-x1+1)
    offset_y = int(round(size*offset_y))
    y1 = y1 + offset_y
    y2 = y1 + size
    x1, y1, x2, y2 = np.array([x1, y1, x2, y2]).round().astype(np.int)
    
    #print((x1, y1, x2, y2), ((x2-x1+1), (y2-y1+1)))
    
    cropped_frames = {} 
    cropped_pts2ds = []
    frame_idxs_ = []
    for _, pts2d, _,  frame_idx in df_fan.values:
        f = frames[frame_idx]
        if pts2d is not None:
            cropped_pts2ds.append(pts2d - (x1, y1))
        else:
            cropped_pts2ds.append(None)
        frame_idxs_.append(frame_idx)
        cropped_frames[frame_idx] = f[y1:y2+1, x1:x2+1].copy()
    df_fan['cropped_pts2d'] = cropped_pts2ds 
    df_fan['cropped_box'] = [np.array([x1, y1, x2, y2])]*len(df_fan)
    df_fan['cropped_size'] = size
    return df_fan, cropped_frames

def save_debug_audio(mp4_path, min_idx, max_idx, audio_path):
    ac = AudioFileClip(mp4_path)
    meta = ff.video_meta(mp4_path)
    s, e = min_idx/meta['nframes'], (max_idx+1)/meta['nframes']
    s, e = s*meta['duration'], e*meta['duration']
    ac = ac.subclip(s, e)
    ac.write_audiofile(audio_path, logger=None)
    
def save_audio(mp4_path, audio_path):
    ac = AudioFileClip(mp4_path)
    ac.write_audiofile(audio_path, logger=None)
    
    
    
def save_crop_info(anchor_box_path, mp4_path, out_dir, make_mp4=False, 
                   crop_offset_y = -0.1, crop_margin=0.4, verbose=False):
    df_anchor_i = pd.read_pickle(anchor_box_path)
    
    # 얼굴이 모두 들어가는 박스 크기를 구한다.
    # 여기서 구한 박스에서만 fan 이 얼굴과 피처 포인트를 구한다.
    box = get_anchor_box(df_anchor_i, offset_y=0, margin=1.0)
    
    min_idx, max_idx = df_anchor_i['frame_idx'].values[[0, -1]]
    
    clip_dir = Path(out_dir)/Path(anchor_box_path).stem
    Path(clip_dir).mkdir(exist_ok=True, parents=True)
    
    try:
        save_audio(mp4_path, f'{clip_dir}/audio.wav')
        save_debug_audio(mp4_path, min_idx, max_idx, f'{clip_dir}/audio_debug.wav')
    except:
        # inference 때는 음성 없는 비디오가 들어온다.
        pass
    
    pickle_path = f'{clip_dir}/df_fan.pickle'
    if Path(pickle_path).exists():
        return pickle_path
    
    frames = ff.extract_frame(mp4_path, min_idx, max_idx+1)
    
    # FAN 이 얼굴과 피처 포인트를 구한다.
    df = df_fan_info(frames, box, verbose=verbose)
    
    # 모델에 입력할 박스를 다시 구해서 crop 한다.
    # crop 박스 영역은 피쳐 포인트 기반으로 구한다.
    df, cropped_frames = crop(frames, df, 
                              offset_y=crop_offset_y, 
                              margin=crop_margin)
    if df is None:
        return None
    
    for (idx1, pts), (idx2, im) in zip(
                                df[['frame_idx','cropped_pts2d']].values, 
                                cropped_frames.items()):
        assert idx1 == idx2
        name = f"""{idx1:05d}_{'yes' if pts is not None else 'no'}.jpg"""
        cv2.imwrite(str(Path(clip_dir)/str(name)), im[:,:,[2,1,0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    df.to_pickle(pickle_path)
    with open(pickle_path.replace('.pickle', '.txt'), 'w') as f:
        f.write('success')
    
    if make_mp4:
        meta = ff.video_meta(mp4_path)
        debug_clip_path = save_debug_clip(clip_dir, meta['fps'])
        print('saved debug_mp4:', debug_clip_path )
        
    return pickle_path
    
    
def save_debug_clip(clip, fps):
    jpgs = glob(f'{clip}/*.jpg')
    jpgs = sorted([(int(Path(e).stem.split('_')[0]), imageio.imread(e)) for e in jpgs])
    
    fan_pts = pd.read_pickle(Path(clip)/'df_fan.pickle')
    fan_pts = fan_pts.set_index('frame_idx')['cropped_pts2d']
    
    def draw_pts(im, pts):
        im = im.copy()
        if pts is not None:
            for (x, y) in pts:
                cv2.circle(im, (x,y), radius=1, color=(0, 255, 0))
        return im
     
    marked = [draw_pts(im, fan_pts[idx]) for idx, im in jpgs]
    merged = [np.concatenate([im, m], axis=1) 
              for (idx, im), m  in zip(jpgs, marked)]
    
    sz = merged[0].shape[0]
    pw = (sz+1)//2*2 - sz
    merged = [np.pad(im, ((0,pw), (0,0), (0,0)), mode='constant', constant_values=128) for im in merged]
    
    audio_clip = AudioFileClip(f'{clip}/audio_debug.wav')
    
    clip_debug = ImageSequenceClip(merged, fps)
    
    clip_debug = clip_debug.set_audio(audio_clip)
    
    save_path = f'{clip}/debug.mp4'
    clip_debug.write_videofile(save_path, logger=None)
    return save_path 


def crop_and_save(path, df_fan, offset_y, margin, clip_dir):
    df_fan = df_fan.copy()
    
    #ToDo: None을 제거해야 됨. crash 발생
    pts2ds = [e for e in df_fan['pts2d'].values if e is not None]
    if len(pts2ds):
        pts2ds = np.stack(pts2ds)
        x1, y1 = pts2ds[:,:,0].min(), pts2ds[:,:,1].min()
        x2, y2 = pts2ds[:,:,0].max(), pts2ds[:,:,1].max()
    else:
        return None, None
    
    cx, cy = (x1+x2)/2, (y1+y2)/2
    sx, sy = (x2-x1+1)*(1+margin), (y2-y1+1)*(1+margin)
    x1, y1 = cx-sx/2, cy-sy/2
    x2, y2 = cx+sx/2, cy+sy/2
    
    
    size = (x2-x1+1)
    offset_y = int(round(size*offset_y))
    y1 = y1 + offset_y
    y2 = y1 + size
    x1, y1, x2, y2 = np.array([x1, y1, x2, y2]).round().astype(np.int)
    
    #print((x1, y1, x2, y2), ((x2-x1+1), (y2-y1+1)))
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()  # meta data, e.g. meta["size"] -> (width, height)
    frame_size = meta['size']
    
    cropped_pts2ds = []
    for (_, pts2d, _,  frame_idx), f in tqdm(zip(df_fan.values, reader), total=len(df_fan), desc='crop_and_save'):
        f = np.frombuffer(f, dtype=np.uint8)
        f = f.reshape(frame_size[1], frame_size[0], 3)
        if pts2d is not None:
            cropped_pts2ds.append(pts2d - (x1, y1))
        else:
            cropped_pts2ds.append(None)
        cropped_frame = f[y1:y2+1, x1:x2+1].copy()
        
        name = f"""{frame_idx:05d}_{'yes' if pts2d is not None else 'no'}.jpg"""
        cv2.imwrite(str(Path(clip_dir)/str(name)), cropped_frame[:,:,[2,1,0]], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
    df_fan['cropped_pts2d'] = cropped_pts2ds 
    df_fan['cropped_box'] = [np.array([x1, y1, x2, y2])]*len(df_fan)
    df_fan['cropped_size'] = size
    return df_fan
    
    
# df_fan_info 와 기능은 동일하고, 메모리 사용량만 줄임
def df_fan_info2(path, box, verbose=False):
    x1, y1, x2, y2 = box
    
    def fan_info(f):
        face = f[y1:y2+2, x1:x2+1]
        pts2d, box = face_detect_fan(face)
        #pts3d, _   = face_detect_fan(face, type3d=True)
        pts3d = None
        return box, pts2d, pts3d
    
    def to_full(box, pts2d, pts3d, x1y1):
        if box is not None:
            box   = (box.reshape(-1,2) + x1y1).reshape(-1)
        if pts2d is not None:
            pts2d = pts2d + x1y1
        if pts3d is not None:
            pts3d = pts3d + (x1y1 +(0,))
        return box, pts2d, pts3d
    
    def __fan_info(f, size):
        f = np.frombuffer(f, dtype=np.uint8)
        f = f.reshape(size[1], size[0], 3)
        return fan_info(f)
    
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()  # meta data, e.g. meta["size"] -> (width, height)
    size = meta["size"]
    
    frame_cnt, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    fi = {idx:__fan_info(frame, size)
          for idx, frame in tqdm(enumerate(reader), total=frame_cnt, desc='■ fan ', disable=not verbose)}
    fi = {idx: to_full(*info, (x1, y1)) for idx, info in fi.items()}
    
    df = pd.DataFrame(fi.values(), columns=['box', 'pts2d', 'pts3d'])
    df['frame_idx'] = list(fi.keys())
    return df
    


# save_crop_info 와 기능은 동일하고, 메모리 사용량을 줄인 것
def save_crop_info2(anchor_box_path, mp4_path, out_dir, make_mp4=False, 
                    crop_offset_y = -0.1, crop_margin=0.4, verbose=False):
    df_anchor_i = pd.read_pickle(anchor_box_path)
    
    # 얼굴이 모두 들어가는 박스 크기를 구한다.
    # 여기서 구한 박스에서만 fan 이 얼굴과 피처 포인트를 구한다.
    box = get_anchor_box(df_anchor_i, offset_y=0, margin=1.0)
    
    min_idx, max_idx = df_anchor_i['frame_idx'].values[[0, -1]]
    
    clip_dir = Path(out_dir)/Path(anchor_box_path).stem
    Path(clip_dir).mkdir(exist_ok=True, parents=True)
    
    try:
        save_audio(mp4_path, f'{clip_dir}/audio.wav')
        save_debug_audio(mp4_path, min_idx, max_idx, f'{clip_dir}/audio_debug.wav')
    except:
        # inference 때는 음성 없는 비디오가 들어온다.
        pass
    
    pickle_path = f'{clip_dir}/df_fan.pickle'
    if Path(pickle_path).exists():
        return pickle_path
    
    # FAN 이 얼굴과 피처 포인트를 구한다.
    df = df_fan_info2(mp4_path, box, verbose=verbose)
    
    # 모델에 입력할 박스를 다시 구해서 crop 한다.
    # crop 박스 영역은 피쳐 포인트 기반으로 구한다.
    df = crop_and_save(mp4_path, df,  offset_y=crop_offset_y,  margin=crop_margin, clip_dir=clip_dir)
    if df is None:
        return None
    df.to_pickle(pickle_path)
    with open(pickle_path.replace('.pickle', '.txt'), 'w') as f:
        f.write('success')
    
    if make_mp4:
        meta = ff.video_meta(mp4_path)
        debug_clip_path = save_debug_clip(clip_dir, meta['fps'])
        print('saved debug_mp4:', debug_clip_path )
        
    return pickle_path