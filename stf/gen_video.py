from pathlib import Path
import os
import torch
import imageio
import gc
import librosa
import soundfile
import random
import shutil
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip
import math
from glob import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import pandas as pd
import cv2
import sys
from .util import *
from .s2f_dir.src.datagen_aug import LipGanDS 
from .preprocess_dir.utils import face_finder as ff
from .preprocess_dir.utils import crop_with_fan as cwf 
from .preprocess_dir.utils import make_mels as mm 
import imageio_ffmpeg 
from joblib import Parallel, delayed
from PIL import Image
import threading
from queue import Queue


# 임시 global 변수 담는 곳
global_v_ = {}

# model inference 를 위해 audio 파일을 준비시킨다. (wav 길이만큼 dummy 이미지 생성, mel 생성)
def audio_crop(wav_std, wav_std_ref_wav, wav_path, crop_wav_dir, video_fps, verbose=False):
    shutil.rmtree(crop_wav_dir, ignore_errors=True)
    Path(crop_wav_dir).mkdir(exist_ok=True, parents=True)
    
    ### 오디오 파일을 mels로 변환한다
    if wav_std and wav_std_ref_wav is not None:
        sr = 22050
        w, sr = librosa.load(wav_path)
        r, sr = librosa.load(wav_std_ref_wav)
        if verbose:
            print('*** np.std(w): ', np.std(w))
        w = (w / np.std(w)) * np.std(r)
        if verbose:
            print('*** np.std(w), np.std(r): ', np.std(w), np.std(r))
        temp_wav = f'{crop_wav_dir}/{Path(wav_path).stem}.{random.randint(0, 10000)}.wav'
        soundfile.write(temp_wav, w, 22050)
        wav_path = temp_wav 
        
    if verbose:
        print('[1/5] 오디오 파일 변환 ... ')
    mels = mm.load_wav_to_mels(wav_path)
    np.savez_compressed(f'{crop_wav_dir}/mels', spec=mels)
    
    # 오디오 클립을 읽어온다
    au = AudioFileClip(wav_path)

    if verbose:
        print('[2/5] 오디오 시간에 맞춰 dummy 이미지 저장 ... ')
    # 오디오 시간만큼 더미 사진을 만든다.
    image_count = math.floor(au.duration * video_fps)
    dummy = np.zeros((64,64), np.uint8)
    for i in range(image_count):
        imageio.imwrite(f'{crop_wav_dir}/{i:05d}_yes.jpg',dummy)

    with open(f'{crop_wav_dir}/fps.txt', 'w') as f:
        f.write(f'{video_fps}')
        
    au.close()
    del au
    return wav_path


# model inference 를 위한 파일 목록을 만들어서 결과를 리턴한다.
def dataset_val_images(template, wav_path, wav_std, wav_std_ref_wav, fps, video_start_offset_frame=None, verbose=False):
    
    template_video_path = template.template_video_path
    VSO = video_start_offset_frame
    assert(VSO is None or VSO >= 0)
    
    crop_wav_dir = os.path.join(template.model.work_root_path, 'temp', template.model.args.name,
                                f'crop_audio_{Path(wav_path).stem}')
    wav_path2 = audio_crop(wav_std, wav_std_ref_wav, wav_path, crop_wav_dir, fps, verbose=verbose)
    
    # 오디오에 대응되는 더미 사진을 준비한다
    audios = sorted(glob(f'{crop_wav_dir}/*.jpg'))
    
    # 비디오 사진도 더미 갯수만큼 준비한다(갯수는 뒤부터 끊는다)
    images = sorted(glob(f'{template.crop_mp4_dir}/{Path(template_video_path).stem}_000/*.jpg'))
    if verbose:
        print('len images:', len(images))
    first_frame_idx = int(Path(images[0]).stem.split('_')[0])
    
    # template video 의 첫번째 이미지는 0 부터 시작해야한다.
    if first_frame_idx != 0:
        raise Exception(f'template video have some error:{template_video_path}, first_frame_idx:{first_frame_idx}')
    
    # audio 가 template video 보다 너무 길면 에러를 낸다.
    if len(images) < len(audios) and (VSO is not None and len(images) < len(audios) + VSO):
        raise Exception('wav is too long than template video')
    
    # template video, audio 중 더 짧은쪽에 맞춰 생성한다.
    len_frame =  min(len(images),len(audios))
    if verbose:
        print('len(images), len(audios), len_frame ', len(images), len(audios), len_frame)
        
    audios = audios[:len_frame]
    #images = images[:len_frame]
    if VSO is None: # 비디오 갯수는 뒤부터 끊는다
        images = images[-len_frame:]
    else: # 비디오 갯수는 지정된 숫자부터 끊는다
        images = images[VSO:VSO+len_frame]
    duration = len_frame/fps
    #print(duration, au.duration)
    au = AudioFileClip(wav_path2)
    assert duration <= au.duration
    assert len(audios) == len(images)
    if verbose:
        print('len(images), len(audios), len_frame ', len(images), len(audios), len_frame)
    au.close()
    del au
        
    return sorted(audios) + sorted(images)


#g_save_image = False
# model inference 한다.
def inference_model(template, val_images, device, callback=None, verbose=False):
    #global g_save_image
    mask_ver = template.model.args.mask_ver
    #args = Dict(
    #    batch_size = 32,
    #    num_workers = 8,
    #    fps = video_fps,
    #    mel_step_size = 108, #81,
    #    mel_ps = 80,
    #    val_images = sorted(audios + images),
    #    img_size = 352,
    #    mask_ver = mask_ver,
    #    num_ips = 2,
    #    mask_img_trsf_ver = 0,
    #    mel_trsf_ver = -1,
    #    mel_norm_ver = -1,
    #    lr = 1, # dummy_lr
    #)
    args = copy.deepcopy(template.model.args)
    args.val_images = val_images
    
    ds = LipGanDS(args, 'val')
    dl = DataLoader(dataset=ds, batch_size=args.batch_size, num_workers=args.num_workers,
                   worker_init_fn=seed_worker)

    outs = []
    def to_img(t):
        img = t.cpu().numpy().astype(np.float64)
        img = ((img / 2.0) + 0.5) * 255.0
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img
    
        
    for idx, (img_gt, mel, ips) in tqdm(enumerate(dl), total=len(dl), desc='inference model', disable=not verbose):
        audio = mel.unsqueeze(1).to(device)
        #if g_save_image:
        #    print()
        #    save_img_before = to_img(ips[0].clone())
        #    print(save_img_before.shape)
        #    Image.fromarray(save_img_before[:,:,:3]).save('./temp_before.png')
        ips = ips.to(device).permute(0, 3, 1, 2)
        with torch.no_grad():
            pred = template.model.model(ips, audio)
        gen_face = to_img(pred.permute(0, 2, 3, 1))
        outs += list(gen_face)
        callback(idx/len(dl)*100)
        #if g_save_image:
        #    Image.fromarray(gen_face[0]).save('./temp_after.png')
        #    g_save_image = False

    # BGR -> RBG
    imgs = [im[:,:,[2,1,0]] for im in outs]


                                         
    del dl
    del ds
    del args
    del outs

    return imgs

g_save_image = False

# generator : model inference 한다. 
def gen_inference_model(template, val_images, device, verbose=False):
    global g_save_image
    mask_ver = template.model.args.mask_ver
    #args = Dict(
    #    batch_size = 32,
    #    num_workers = 8,
    #    fps = video_fps,
    #    mel_step_size = 108, #81,
    #    mel_ps = 80,
    #    val_images = sorted(audios + images),
    #    img_size = 352,
    #    mask_ver = mask_ver,
    #    num_ips = 2,
    #    mask_img_trsf_ver = 0,
    #    mel_trsf_ver = -1,
    #    mel_norm_ver = -1,
    #    lr = 1, # dummy_lr
    #)
    args = copy.deepcopy(template.model.args)
    args.val_images = val_images
    
    ds = LipGanDS(args, 'val')
    dl = DataLoader(dataset=ds, batch_size=args.batch_size, num_workers=args.num_workers,
                   worker_init_fn=seed_worker)

    outs = []
    def to_img(t):
        img = t.cpu().numpy().astype(np.float64)
        img = ((img / 2.0) + 0.5) * 255.0
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img
    
        
    for idx, (img_gt, mel, ips) in enumerate(dl):
        audio = mel.unsqueeze(1).to(device)
        if g_save_image:
            save_img_before = to_img(ips[0].clone())
            print(save_img_before.shape)
            Image.fromarray(save_img_before[:,:,:3]).save('./temp_before.png')
        ips = ips.to(device).permute(0, 3, 1, 2)
        with torch.no_grad():
            pred = template.model.model(ips, audio)
        gen_face = to_img(pred.permute(0, 2, 3, 1))
        # BGR -> RBG
        gen_face = gen_face[:,:,[2,1,0]]
        gen_face = list(gen_face)
        for o in gen_face:
            yield o
        outs += gen_face
        if g_save_image:
            Image.fromarray(gen_face[0]).save('./temp_after.png')
            g_save_image = False

                                         
    del dl
    del ds
    del args


# template video 의 frame 과 model inference 결과를 합성한다.
def compose(template, model_outs, video_start_offset_frame, full_imgs, n_compose_worker_num=5, verbose=False):
    imgs = model_outs
    args = template.model.args
    VSO = video_start_offset_frame
    first_frame_idx = 0
    
    if verbose:
        print('[5/5] 비디오 합성 ... ')
    df = pd.read_pickle(f'{template.crop_mp4_dir}/{Path(template.template_video_path).stem}_000/df_fan.pickle')
    #sz = df['cropped_size'].values[0]
    x1, y1, x2, y2 = df['cropped_box'].values[0].round().astype(np.int)
    del df
    
    img_size = args.img_size
    if verbose:
        print(x2-x1+1, y2-y1+1, imgs[0].shape)
    
    # resize
    inter_alg = cv2.INTER_AREA if x2-x1+1 < img_size else cv2.INTER_CUBIC 
    if verbose:
        print('n_compose_worker_num:', n_compose_worker_num)
        print('resize:', 'INTER_AREA' if inter_alg == cv2.INTER_AREA else 'INTER_CUBIC')
    imgs = [cv2.resize(c, (x2-x1+1, y2-y1+1), inter_alg) for c in tqdm(imgs, desc='resize to original crop', disable=not verbose)]
    
    def get_mask(width, height, gradation_width=50):
        mask = np.ones((height, width, 1))
        r = list(range(0,gradation_width,1))
        for s, e in zip(r, r[1:]):
            g = s / gradation_width
            #print(f'---- s:{s}, e:{e}, g:{g}')
            mask[s:e,               s:width-s,       :] = g
            mask[height-e:height-s, s:width-s,       :] = g
            mask[s:height-s,        s:e,             :] = g
            mask[s:height-s,        width-e:width-s, :] = g
        return mask
    
    mask = get_mask(x2-x1+1,y2-y1+1,30)
    mask_crop = mask
    mask_origin = (mask - 1) * -1    
    
    #full_imgs = template.full_frames[VSO+first_frame_idx : VSO + len(imgs)+first_frame_idx]
    
    # 음성과 video sync 를 위해 적당히 정해진 crop_start_frame 만큼 잘라낸다.
    crop_start_frame = template.model.args.crop_start_frame
    if crop_start_frame >= 0:
        full_imgs2 = full_imgs[crop_start_frame:] + ([full_imgs[-1]]*crop_start_frame)
        imgs = imgs[crop_start_frame:] + ([imgs[-1]]*crop_start_frame)
    else:
        full_imgs2 = ([full_imgs[0]]*abs(crop_start_frame)) + full_imgs[:crop_start_frame]
        imgs = ([imgs[0]]*abs(crop_start_frame)) + imgs[:crop_start_frame]
    assert(len(full_imgs) == len(full_imgs2))
    assert(len(imgs) == len(imgs))
    
    sz_x = x2-x1+1
    sz_y = y2-y1+1
        
    #composed = []
    #for c, f in tqdm(zip(imgs, full_imgs2), total=len(imgs), desc='compose original frames and model outputs', disable=not verbose):
    #    composed.append(f.copy())
    #    composed[-1][y1:y2+1, x1:x2+1] = (f[y1:y2+1, x1:x2+1] * mask_origin + c[0:sz_y, 0:sz_x] * mask_crop)# [:,:-2,:]
    
    def compose_func(v):
        c, f = v
        frame = f.copy()
        frame[y1:y2+1, x1:x2+1] = (f[y1:y2+1, x1:x2+1] * mask_origin + c[0:sz_y, 0:sz_x] * mask_crop)# [:,:-2,:]
        return frame
    composed = Parallel(n_jobs=n_compose_worker_num)(
        delayed(compose_func)((c, f))
        for c, f in tqdm(zip(imgs, full_imgs2), total=len(imgs),
                         desc='compose original frames and model outputs',
                         disable=not verbose)
    )
    return composed


# 음성과 video sync 를 위해 적당히 정해진 crop_start_frame 만큼 잘라낸다.
# 길이를 유지하기 위해 잘라낸 것 만큼 동일 이미지를 반복한다.
def crop_start_frame(images, crop_cnt):
    if crop_cnt >= 0:
        images2 = images[crop_cnt:] + ([images[-1]]*crop_cnt)
    else:
        images2 = ([images[0]]*abs(crop_cnt)) + images[:crop_cnt]
    assert(len(images) == len(images2))
    return images2


# 음성과 video sync 를 위해 적당히 정해진 crop_start_frame 만큼 잘라낸다.

# 비디오 쓰기(composed 이미지들을 하나의 video 로 만든다.)
def write_video(composed, wav_path, fps, output_path, slow_write, verbose=False):
    duration = len(composed)/fps
    
    ac = AudioFileClip(wav_path)
    if verbose:
        print(ac.duration, duration, abs(ac.duration- duration))
    assert(abs(ac.duration - duration) < 0.1)
    
    clip = ImageSequenceClip(composed, fps=fps)
    clip = clip.set_audio(ac.subclip(ac.duration-duration, ac.duration))
    h, w, _ = composed[0].shape
    if h > 1920:
        clip = clip.resize((w//2, h//2))

    ffmpeg_params = None
    if slow_write:
        ffmpeg_params=['-acodec', 'aac', '-preset', 'veryslow', '-crf', '17']
        
    temp_out = output_path
    Path(temp_out).parent.mkdir(exist_ok=True)
    if verbose:
        clip.write_videofile(temp_out, ffmpeg_params=ffmpeg_params)
    else:
        clip.write_videofile(temp_out, ffmpeg_params=ffmpeg_params, verbose=verbose, logger=None)
    
    clip.close()
    ac.close()
    del clip
    del ac
    

# model inference, template video frame와 inference 결과 합성, 비디오 생성 작업을 한다.
def gen_video(template, wav_path, wav_std, wav_std_ref_wav,
              video_start_offset_frame, out_path, full_imgs,
              n_compose_worker_num=5,
              head_only=False, slow_write=True, verbose=False):
        
    device = template.model.device
    fps = template.fps
    
    # model inference 를 위한 데이터 준비
    val_images = dataset_val_images(template, wav_path, wav_std, wav_std_ref_wav,
                                    fps=fps,
                                    video_start_offset_frame=video_start_offset_frame,
                                    verbose=verbose)
    if verbose:
        print('len(val_images) : ', len(val_images))

    # model inference
    outs = inference_model(template, val_images, device, verbose=verbose)

    if head_only:
        composed = crop_start_frame(outs, template.model.args.crop_start_frame)
    else:
        # template video 와 model inference 결과 합성
        composed = compose(template, outs, video_start_offset_frame, full_imgs,
                           n_compose_worker_num=n_compose_worker_num,
                           verbose=verbose)
    
    # 비디오 생성
    write_video(composed, wav_path, fps, out_path, slow_write, verbose=verbose)

    del composed
    del outs
    del val_images
    gc.collect()
    
    return out_path
    

# model inference, template video frame와 inference 결과 합성, 비디오 생성 작업을 한다.
def gen_video2(template, wav_path, wav_std, wav_std_ref_wav,
               video_start_offset_frame, out_path, full_imgs,
               n_compose_worker_num=5,
               head_only=False, slow_write=True, verbose=False):
        
    device = template.model.device
    fps = template.fps
    
    # model inference 를 위한 데이터 준비
    val_images = dataset_val_images(template, wav_path, wav_std, wav_std_ref_wav,
                                    fps=fps,
                                    video_start_offset_frame=video_start_offset_frame,
                                    verbose=verbose)
    if verbose:
        print('len(val_images) : ', len(val_images))

    # model inference
    outs = inference_model(template, val_images, device, verbose=verbose)
    
    # crop_start_frame 만큼 잘라낸다.
    outs = crop_start_frame(outs, template.model.args.crop_start_frame)
    full_imgs = crop_start_frame(full_imgs, template.model.args.crop_start_frame)
        
    # 함성함수를 얻는다.
    compose_func = get_compose_func(template, verbose=verbose)
    write_video2(out_path=out_path,
                 compose_func=compose_func, 
                 model_out=outs,
                 full_frames=full_imgs,
                 wav_path=wav_path,
                 fps=fps,
                 output_path=out_path, 
                 slow_write=slow_write,
                 n_compose_worker_num=n_compose_worker_num,
                 verbose=verbose)
    del outs
    del val_images
    gc.collect()
    
    return out_path


def write_video2(out_path, compose_func, model_out, full_frames,
                 wav_path, fps, output_path, slow_write,
                 n_compose_worker_num=5,
                 verbose=False):
    if verbose:
        print('[5/5] 비디오 합성 함수... ')
        print(f"save video {out_path}")
    
    # 합성하면서 비디오 생성
    ffmpeg_params = None
    if slow_write:
        ffmpeg_params=['-acodec', 'aac', '-preset', 'veryslow', '-crf', '17']

    # document : https://github.com/imageio/imageio-ffmpeg
    size = full_frames[0].shape[:2][::-1]
    writer = imageio_ffmpeg.write_frames(out_path,
                                         size = size,
                                         fps=fps,
                                         ffmpeg_log_level='debug',
                                         quality = 10, # 0~10
                                         output_params=ffmpeg_params,
                                         audio_path = wav_path, )
    
    writer.send(None)  # seed the generator
    out_memory_shape = full_frames[0].shape
    for o, f in tqdm(zip(model_out, full_frames), total=len(model_out), desc="compose (model out, template)", disable=not verbose):
        frame = compose_func(o, f)
        writer.send(frame)  # seed the generator
    writer.close()
    
    
    #composed = Parallel(n_jobs=n_compose_worker_num)(
    #    delayed(compose_func)((c, f))
    #    for c, f in tqdm(zip(imgs, full_imgs2), total=len(imgs),
    #                     desc='compose original frames and model outputs',
    #                     disable=not verbose)
    #)
    
    
# template video 의 frame 과 model inference 결과를 합성한다.
def get_compose_func(template, verbose=False):
    args = template.model.args
    
    df = pd.read_pickle(f'{template.crop_mp4_dir}/{Path(template.template_video_path).stem}_000/df_fan.pickle')
    #sz = df['cropped_size'].values[0]
    x1, y1, x2, y2 = df['cropped_box'].values[0].round().astype(np.int)
    del df
    
    img_size = args.img_size
    if verbose:
        print('croped size: ', x2-x1+1, y2-y1+1)
        print('croped region(x1,y1,x2,y2): ', x1, y1, x2, y2)
        
    def get_mask(width, height, gradation_width=50):
        mask = np.ones((height, width, 1))
        r = list(range(0,gradation_width,1))
        for s, e in zip(r, r[1:]):
            g = s / gradation_width
            #print(f'---- s:{s}, e:{e}, g:{g}')
            mask[s:e,               s:width-s,       :] = g
            mask[height-e:height-s, s:width-s,       :] = g
            mask[s:height-s,        s:e,             :] = g
            mask[s:height-s,        width-e:width-s, :] = g
        return mask
    
    gradation_width = int((y2-y1)*0.1)
    if verbose:
        print('gradation_width : ', gradation_width)
    mask = get_mask(x2-x1+1,y2-y1+1,gradation_width)
    mask_crop = mask
    mask_origin = (mask - 1) * -1    
    
    sz_x = x2-x1+1
    sz_y = y2-y1+1
        
    # resize algorithm
    inter_alg = cv2.INTER_AREA if x2-x1+1 < img_size else cv2.INTER_CUBIC 
    
    def compose_one(model_out, full_img):
        # 1. resize
        img = cv2.resize(model_out, (x2-x1+1, y2-y1+1), inter_alg)
        # 2. 붙여넣기
        c, f = img, full_img
        out_memory = full_img.copy()
        out_memory[y1:y2+1, x1:x2+1] = (f[y1:y2+1, x1:x2+1] * mask_origin + c[0:sz_y, 0:sz_x] * mask_crop)# [:,:-2,:]
        return out_memory
    
    return compose_one


# model inference, template video frame와 inference 결과 합성, 비디오 생성 작업을 한다.
def gen_video3(template, wav_path, wav_std, wav_std_ref_wav,
               video_start_offset_frame, out_path,
               head_only=False,
               slow_write=True,
               callback=None,
               verbose=False):
    device = template.model.device
    fps = template.fps
        
    callback1 = callback_inter(callback, min_per=0, max_per=30,
                               desc='gen_video3 1', verbose=verbose)
    callback2 = callback_inter(callback, min_per=30, max_per=100,
                               desc='gen_video3 2', verbose=verbose)
    
    # model inference 를 위한 데이터 준비
    val_images = dataset_val_images(template, wav_path, wav_std, wav_std_ref_wav,
                                    fps=fps,
                                    video_start_offset_frame=video_start_offset_frame,
                                    verbose=verbose)
    if verbose:
        print('len(val_images) : ', len(val_images))

    # model inference
    outs = inference_model(template, val_images, device, callback=callback1, verbose=verbose)
    
    # crop_start_frame 만큼 잘라낸다.
    outs = crop_start_frame(outs, template.model.args.crop_start_frame)

    # 함성함수를 얻는다.
    compose_func = get_compose_func(template, verbose=verbose)
    write_video3(out_path=out_path,
                 compose_func=compose_func, 
                 model_out=outs,
                 template_video_path=template.template_video_path,
                 wav_path=wav_path,
                 fps=fps,
                 crop_start_frame_count=template.model.args.crop_start_frame,
                 output_path=out_path, 
                 slow_write=slow_write,
                 callback=callback2,
                 verbose=verbose)
    del outs
    del val_images
    gc.collect()
    callback2(100)
    
    return out_path


def write_video3(out_path, compose_func, model_out, template_video_path,
                 wav_path, fps, crop_start_frame_count,
                 output_path, slow_write,
                 callback=None,
                 verbose=False):
    if verbose:
        print('[5/5] 비디오 합성 함수... ')
        print(f"save video {out_path}")
    
    # 합성하면서 비디오 생성
    ffmpeg_params = None
    if slow_write:
        ffmpeg_params=['-acodec', 'aac', '-preset', 'veryslow', '-crf', '17']

    # document : https://github.com/imageio/imageio-ffmpeg
    reader = imageio_ffmpeg.read_frames(template_video_path)
    meta = reader.__next__()  # meta data, e.g. meta["size"] -> (width, height)
    size = meta["size"]
    if verbose:
        print(meta)
    assert(crop_start_frame_count >= 0)
    
    # crop_start_frame_count 만큼 앞에서 버린다.
    if crop_start_frame_count > 0:
        # TODO snow : 왜인지 모르겠지만, crop_start_frame_count-1 로 해야지만 프레임이 맞는다.
        for f, i in zip(reader, range(crop_start_frame_count-1)):
            #print('crop:', i)
            pass
    
    writer = imageio_ffmpeg.write_frames(out_path,
                                         size = size,
                                         fps=fps,
                                         ffmpeg_log_level='error',
                                         quality = 10, # 0~10
                                         output_params=ffmpeg_params,
                                         audio_path = wav_path, )
    
    writer.send(None)  # seed the generator
    try:
        for idx, (o, f, _) in tqdm(enumerate(zip(model_out, reader, range(len(model_out)-crop_start_frame_count))),
                         total=len(model_out), desc="compose (model out, template)",
                         disable=not verbose):
            f = np.frombuffer(f, dtype=np.uint8)
            #print(f.shape, size, f.dtype)
            f = f.reshape(size[1], size[0], 3)
            frame = compose_func(o, f)
            writer.send(frame)  # seed the generator
            callback(idx/len(model_out)*100)
        for _ in tqdm(range(crop_start_frame_count),
                      total=crop_start_frame_count, desc="compose (model out, template)",
                      disable=not verbose):
            f = np.frombuffer(f, dtype=np.uint8)
            #print(f.shape, size, f.dtype)
            f = f.reshape(size[1], size[0], 3)
            frame = compose_func(o, f)
            writer.send(frame)  # seed the generator
            callback(idx/len(model_out)*100)
    except Exception as e:
        print('exception read template video stream', e)
        raise Exception('except than template video', e)
        pass
        
    writer.close()


def write_video_in_thread(out_path, compose_func, task_name, template_video_path,
                 wav_path, fps, crop_start_frame_count, total_cnt,
                 output_path, slow_write,
                 verbose=False):
    global global_v_
    
    if verbose:
        print('[5/5] 비디오 합성 함수... ')
        print(f"save video {out_path}")
    
    # 합성하면서 비디오 생성
    ffmpeg_params = None
    if slow_write:
        ffmpeg_params=['-acodec', 'aac', '-preset', 'veryslow', '-crf', '17']

    # document : https://github.com/imageio/imageio-ffmpeg
    reader = imageio_ffmpeg.read_frames(template_video_path)
    meta = reader.__next__()  # meta data, e.g. meta["size"] -> (width, height)
    size = meta["size"]
    if verbose:
        print(meta)
    assert(crop_start_frame_count >= 0)
    
    # crop_start_frame_count 만큼 앞에서 버린다.
    if crop_start_frame_count > 0:
        # 개수를 조정해준다.
        total_cnt = total_cnt - crop_start_frame_count
        # model output 버린다.
        for i in range(crop_start_frame_count):
            global_v_[task_name].get()
            #print('crop:', i)
            pass
        # TODO snow : 왜인지 모르겠지만, crop_start_frame_count-1 로 해야지만 프레임이 맞는다.
        for f, i in zip(reader, range(crop_start_frame_count-1)):
            #print('crop:', i)
            pass
    
    writer = imageio_ffmpeg.write_frames(out_path,
                                         size = size,
                                         fps=fps,
                                         ffmpeg_log_level='error',
                                         quality = 10, # 0~10
                                         output_params=ffmpeg_params,
                                         audio_path = wav_path, )
    
    writer.send(None)  # seed the generator
    try:
        for idx, (f, _) in tqdm(enumerate(zip(reader, range(total_cnt-crop_start_frame_count))),
                         total=total_cnt, desc="compose (model out, template)",
                         disable=not verbose):
            o = global_v_[task_name].get()
            if o is None:
                print('alread None??!!!')
                break
            #print('worker:', idx, ', th:', threading.get_ident())
            
            f = np.frombuffer(f, dtype=np.uint8)
            #print(f.shape, size, f.dtype)
            f = f.reshape(size[1], size[0], 3)
            frame = compose_func(o, f)
            writer.send(frame)  # seed the generator
        for _ in tqdm(range(crop_start_frame_count),
                      total=crop_start_frame_count, desc="compose (model out, template)",
                      disable=not verbose):
            f = np.frombuffer(f, dtype=np.uint8)
            #print(f.shape, size, f.dtype)
            f = f.reshape(size[1], size[0], 3)
            frame = compose_func(o, f)
            writer.send(frame)  # seed the generator
    except Exception as e:
        print('exception read template video stream', e)
        raise Exception('except than template video', e)
        pass
        
    writer.close()
    

def get_task_name():
    global global_v_
    while 1:
        task_name = random.randint(0, 10000000) # 동시에 여러개 하더라도 각자 할 수 있도록 이름을 준다.
        if task_name in global_v_:
            print('!!!!! exist task, task_names:', global_v_.keys())
            continue
        else:
            return task_name

    
# model inference, template video frame와 inference 결과 합성, 비디오 생성 작업을 한다.
# 속도 개선 : inference, write_video 동시에..
def gen_video4(template, wav_path, wav_std, wav_std_ref_wav,
               video_start_offset_frame, out_path,
               head_only=False,
               slow_write=True,
               callback=None,
               verbose=False):
    global global_v_
    callback_ = callback_inter(callback, min_per=0, max_per=30,
                               desc='gen_video3 1', verbose=verbose)
    
    device = template.model.device
    fps = template.fps
        
    # model inference 를 위한 데이터 준비
    val_images = dataset_val_images(template, wav_path, wav_std, wav_std_ref_wav,
                                    fps=fps,
                                    video_start_offset_frame=video_start_offset_frame,
                                    verbose=verbose)
    if verbose:
        print('len(val_images) : ', len(val_images))

    # model inference
    gen_infer = gen_inference_model(template, val_images, device, verbose=verbose)
        
    task_name = get_task_name()
    global_v_[task_name] = Queue() # 작업큐
    
    # 함성함수를 얻는다.
    compose_func = get_compose_func(template, verbose=verbose)

    # 전체 inference 이미지 개수를 얻는다.
    total_cnt = int(len(val_images)/2)

    # write video worker 실행
    args = (out_path, compose_func, task_name, template.template_video_path,
            wav_path, fps, template.model.args.crop_start_frame, total_cnt,
            out_path, slow_write, verbose)
    thread = threading.Thread(target=write_video_in_thread, args=args)
    thread.start()
    
    # model inference
    for i, v in enumerate(gen_infer):
        global_v_[task_name].put(v)
        # TODO snow : 원래는 queue에 몇개가 소진되었나로 해야하지만.. 일단 이렇게 한다.
        process_cnt = i + 1 - global_v_[task_name].qsize()
        callback_(process_cnt*100/total_cnt) 
    global_v_[task_name].put(None) # 이제 끝났음을 알림
    
    # write video worker 완료 대기
    thread.join()
    callback_(100)
    
    del val_images
    gc.collect()

    global_v_[task_name] = None
    
    return out_path

