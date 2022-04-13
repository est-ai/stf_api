import os
import torch
import torchvision
import imageio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
from moviepy.editor import AudioFileClip, ImageSequenceClip
import cv2
import gc
import imageio_ffmpeg
from stf.util import callback_inter


g_mtcnn = None
g_recognizer = None
g_device = None


# 얼굴 인식 툴킷
def init_face_finder(device='cuda:0'):
    global g_mtcnn
    global g_recognizer
    global g_device

    if g_mtcnn is None and g_recognizer is None:
        g_mtcnn = MTCNN(image_size=166, device=device)
        print('load MTCNN ', 'success ^ ^' if g_mtcnn is not None else 'fail ㅠㅠ')
        g_recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print('load g_recognizer ', 'success ^ ^' if g_recognizer is not None else 'fail ㅠㅠ')
        g_device = device


def del_face_finder():
    global g_mtcnn
    global g_recognizer
    global g_device    
    if g_mtcnn is not None:
        del g_mtcnn
        g_mtcnn = None
    if g_recognizer is not None:
        del g_recognizer
        g_recognizer = None
    torch.cuda.empty_cache()


def find_face(img):
    """ 얼굴 위치 및 임베딩 벡터 구하기 
        Arguments:
            img: torch.Tensor 또는 ndarray. 분석하고자 하는 사진
        동작:
            img 사진에 등장하는 모든 얼굴을 찾아서 embedding값을 구한다
            얼굴 영역 box와 embeddig 값을 pandas.DataFrame 형태로 변환한다
            df와 df의 정보값에 대응되는 crop 영역도 함께 리턴한다.
    """ 
    global g_mtcnn
    global g_recognizer

    # snow : init_face_finder 을 미리 불러주지 않았으면 여기서 불리도록한다.
    init_face_finder()

    if isinstance(img, str):
        img = imageio.imread(img)
    frame = np.array(img)
    df_non_face = pd.DataFrame({'box': [np.nan], 'ebd': [np.nan]})
    with torch.no_grad():
        boxes = g_mtcnn.detect(frame)
        if boxes[0] is None:
            return df_non_face, None
        boxes = boxes[0].round().astype(np.int)

    org = np.array(frame)

    def calc_ebd(box):
        x1, y1, x2, y2 = box
        crop = org[y1:y2 + 1, x1:x2 + 1]
        sz = g_mtcnn.image_size
        resized = cv2.resize(crop, (sz, sz), cv2.INTER_AREA)
        x = torchvision.transforms.functional.to_tensor(resized)
        with torch.no_grad():
            ebd = g_recognizer(x.unsqueeze(0).to(g_device))
        return ebd[0].cpu(), crop
    
    def check_box(x1, y1, x2, y2):
        return (0 <= x1 and 0 <= y1) and (x2 < frame.shape[1] and y2 < frame.shape[0])
    
    boxes = [box.tolist() for box in boxes if check_box(*box)]
    ebds = [calc_ebd(box) for box in boxes]
    if len(ebds) == 0:
        return df_non_face, None
    ebds, face_images = list(zip(*ebds))
    df_face = pd.DataFrame({'box':list(boxes), 'ebd':ebds})
    return df_face, face_images


""" 주어진 비디오에서 얼굴을 찾아 아나운서 얼굴과 유사도 구해 놓기 """

# 비디오에서 추출 랜던 가능한 프레임 범위중 end 부분 알아내기
def get_valid_end(path, end=None, stride=1):
    vid = imageio.get_reader(path, 'ffmpeg')
    
    if end is None:
        end = vid.count_frames() 
    elif end < 0:
        end = vid.count_frames() + 1 + end
    
    if stride == 1:
        return end
        
    try:
        vid.get_data(end - 1)
        vid.close()
        return end 
    except:
        end = end - 1
        vid.close()
        return  get_valid_end(path, end, stride)
    
def extract_frame(path, start=0, end=-1, stride=1, verbose=False):
    val_end = get_valid_end(path, end, stride)
    
    vid = imageio.get_reader(path, 'ffmpeg')
    if end < 0:
        end =  val_end + 1 + end
    if val_end < end:
        end = val_end
           
    frames = {} 
    for i in tqdm(range(start, end, stride), desc=f'extract frame stride({stride}) {Path(path).name}', disable=not verbose):
        try:
            f = vid.get_data(i) 
        except:
            w, h = vid.get_meta_data()['size']
            f = np.zeros((h,w,3), np.uint8)
        frames[i] = f
        
    vid.close()
    return frames
    

# 비디오에 나오는 얼굴 임베딩값 구하는 유틸
def calc_ebds_from_images(frames, verbose=False):
    face_infos = {idx:find_face(frame)[0] for idx, frame in tqdm(frames.items(), desc='find_faces for calc_ebd', disable=not verbose)}
    for idx, fi in face_infos.items():
        fi['frame_idx'] = idx 
    return pd.concat(face_infos, ignore_index=True)
# 유사도 구하는 유틸

# 얼굴 박스 그려서 보여주기.  다른 사람 얼굴은 붉은색,  아나운서 얼굴은 녹색
def draw_face(df, frame):
    frame = frame.copy()
    
    boxes = df['box'].values
    if 1 < len(boxes):
        for x1, y1, x2, y2 in boxes[:-1]:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    if 0 < len(boxes):
        x1, y1, x2, y2 = boxes[-1] 
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return frame

def show_face(frame_idx, df_face_info, frames):
    df = df_face_info.query('frame_idx == @frame_idx')
    frame = draw_face(df, frames[frame_idx])
    display(Image.fromarray(frame))

def get_filtered_face(df_face_info, sim_th=0.7):
    
    # 아나운서 얼굴만 나오는 정사각형 영역 구하기
    tqdm.pandas()
    
    # 유사도 기반으로 아나운서 얼굴만 필터(실제로는 먼저 가장 유사한 얼굴만 골라내기)
    df = df_face_info.groupby('frame_idx', as_index=False).apply(lambda df: df.iloc[-1:])
    df = df.drop('ebd', axis=1)
    df['area'] = df['box'].map(lambda x:(x[2]-x[0]+1)*(x[3]-x[1]+1))
    df = df.query('@sim_th <= similaraty')   
    return df
    
def get_face_info_(frames, ebd_아나운서, sim_th, verbose=False):
    df_face_info = calc_ebds_from_images(frames, verbose=verbose)
    df_face_info = df_face_info.dropna(axis=0)

    calc_sim = lambda ebd: (ebd_아나운서 * ebd).sum().item()
    df_face_info['similaraty'] = df_face_info['ebd'].map(calc_sim)
    df_face_info = df_face_info.sort_values(['frame_idx', 'similaraty'])
    
    # 유사도 기반으로 아나운서 얼굴만 필터(실제로는 먼저 가장 유사한 얼굴만 골라내기)
    return frames, df_face_info, get_filtered_face(df_face_info, sim_th)

def get_face_info(path, ebd_아나운서, start=0, end=-1, stride=1, sim_th=0.7, verbose=False):
    frames = extract_frame(path, start, end, stride, verbose=verbose)
    return get_face_info_(frames, ebd_아나운서,  sim_th, verbose=verbose)


def get_face_idxs(mp4_path, meta):
    STEP_SECONDS = 1
    S = STEP_SECONDS
    
    pickle_path = f'df_face_info/{Path(mp4_path).stem}.pickle'
    df_face_info = pd.read_pickle(pickle_path)
    df_f = get_filtered_face(df_face_info, 0.7)
    
    idxs = sorted(df_f['frame_idx'].tolist())
    
    fps = meta['fps']
    
    start_idxs = [max(int(idxs[0]-S*fps+1), 0)]
    end_idxs   = []
    
    prev_idx = start_idxs[-1]
    for idx in idxs:
        if prev_idx +  fps * 10 < idx:
            end_idxs.append(int(prev_idx+fps*S-1))
            start_idxs.append(int(idx-fps*S+1))
        prev_idx = idx
    end_idxs.append(get_valid_end(mp4_path))
        
    return list(zip(start_idxs, end_idxs)) 

def split(mp4_path, ebd_아나운서, start, end, audioclip, meta):
    frames_i, df_face_info_i, df_f_i = get_face_info(mp4_path, ebd_아나운서, start, end, sim_th=0.7)

    idxs = df_f_i['frame_idx']
    start, end = idxs.min(), idxs.max()
    
    frames_i = {i:f for i, f in frames_i.items() if start <= i  and i <= end}
    
    s, e =  start/meta['nframes'], end/meta['nframes']
    
    if audioclip is not None:
        t = audioclip.duration
        a = audioclip.subclip(t_start=t*s, t_end=t*e)
        c = ImageSequenceClip(list(frames_i.values()), fps=meta['fps'])
        
        c = c.set_audio(a)
    else:
        c = None
    
    return c, df_face_info_i, df_f_i

def save_splited_face_info(mp4_path, ebd_아나운서, save_clip=False):
    meta = video_meta(mp4_path)
    
    audioclip = AudioFileClip(mp4_path) if save_clip else None

    out_paths =[]
    for i, (s, e) in enumerate(get_face_idxs(mp4_path, meta)):
        c = extract_frame(mp4_path, s, e)
        s, e =  np.array(list(c.keys()))[[0, -1]]
        e += 1
        clip, df_face_info_i, df_f_i = split(mp4_path, ebd_아나운서, s, e, audioclip, meta)
        #df_face_info_i.to_pickle(f'df_face_info_i/{Path(mp4_path).stem}_{i:03d}.pickle')
        out_path = f'df_anchor_i/{Path(mp4_path).stem}_{i:03d}.pickle'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_f_i.to_pickle(out_path)
        out_paths.append(out_path)
        if save_clip:
            video_name = f'clip/{Path(mp4_path).stem}_{i:03d}.mp4'
            os.makedirs(os.path.dirname(video_name), exist_ok=True)
            clip.write_videofile(video_name)
    return out_paths
		
def save_face_info(mp4_path, ebd_아나운서, base='./df_face_info'):
    pickle_path = f'{base}/{Path(mp4_path).stem}.pickle'
    
    if not Path(pickle_path).exists():
        fps = video_meta(mp4_path)['fps']
        r = get_face_info( mp4_path, ebd_아나운서, 0, -1, stride=(round(fps)*1))
        frames, df_face_info, df_아나운서_only  = r
		
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        df_face_info.to_pickle(pickle_path)
    
    return save_splited_face_info(mp4_path, ebd_아나운서)     


def face_info_to_anchor(df, stride, val_end=None):
    if val_end is None:
        last_idx = df['frame_idx'].max()
        val_end = last_idx
    rows = []
    for idx in range(val_end+1):
        target_idx = idx//stride *stride
        df_search = df.query('frame_idx == @target_idx')
        assert(len(df_search) > 0)
        box, _, _, sim = df_search.iloc[0].values
        x1, y1, x2, y2 = box
        rows.append([box, idx, sim, (x2-x1)*(y2-y1)])
    
    df_face_info = pd.DataFrame(rows, columns=['box', 'frame_idx', 'sililaraty', 'area'])    
    #df_face_info.head()
    return df_face_info


def save_face_info2(mp4_path, ebd_아나운서, base='./', verbose=False):
    df_face_info_path = os.path.join(base,'df_face_info', f"{str(Path(mp4_path).stem)}.pickle")
    if verbose:
        print('save_face_info2 - df_face_info: ', str(df_face_info_path))

    fps = video_meta(mp4_path)['fps']
    stride = round(fps)*1
    
    if not Path(df_face_info_path).exists():
        r = get_face_info(mp4_path, ebd_아나운서, 0, -1, stride=stride, verbose=verbose)
        frames, df_face_info, df_아나운서_only  = r
        del frames
        gc.collect()
        os.makedirs(os.path.dirname(df_face_info_path), exist_ok=True)
        df_face_info.to_pickle(df_face_info_path)
        
    dst = Path(base) / 'df_anchor_i' / f"{Path(df_face_info_path).stem}_000.pickle"
    if verbose:
        print('df_anchor_i:', str(dst))
    if not Path(dst).exists():      
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        df = pd.read_pickle(df_face_info_path)
        df_ = df.sort_values('similaraty', ascending=False).drop_duplicates(['frame_idx'])
        df_ = df_.query('similaraty >= 0.3')
        #display(df_.groupby('frame_idx').count())
        #pdb.set_trace()
        df_face_info = face_info_to_anchor(df_, stride=stride, val_end=None)
        df_face_info.to_pickle(dst)
        return [dst]
    return [dst]


# 메타데이터 추출 유틸
def video_meta(file):
    vid = imageio.get_reader(file, 'ffmpeg')
    meta = vid.get_meta_data()
    meta['path'] = file 
    meta['nframes'] = vid.count_frames()
    vid.close()
    return meta


# 비디오에 나오는 얼굴 임베딩값 구하는 유틸
def calc_ebds_from_images2(path, stride, callback=None, verbose=False):
    if verbose:
        print('calc_ebds_from_images2, ', path)
    def __find_face(f, size):
        f = np.frombuffer(f, dtype=np.uint8)
        f = f.reshape(size[1], size[0], 3)
        return find_face(f)[0]
    
    reader = imageio_ffmpeg.read_frames(path)
    meta = reader.__next__()  # meta data, e.g. meta["size"] -> (width, height)
    size = meta["size"]
    
    frame_cnt, _ = imageio_ffmpeg.count_frames_and_secs(path)
    face_infos = {}
    for idx, frame in tqdm(enumerate(reader), total=frame_cnt, desc='find_faces for calc_ebd', disable=not verbose):
        # 진행상황을 알려준다.
        callback((idx+1)/frame_cnt*100)
            
        if idx % stride != 0:
            continue
        face_infos[idx] = __find_face(frame, size)
        
    for idx, fi in face_infos.items():
        fi['frame_idx'] = idx 
    return pd.concat(face_infos, ignore_index=True)


# get_face_info 와 기능은 동일하지만, 메모리 사용을 줄인 버전
def get_face_info2(path, ebd_아나운서, stride=1, sim_th=0.7, callback=None, verbose=False):
    if verbose:
        print('get_face_info2')
    df_face_info = calc_ebds_from_images2(path, stride=stride, callback=callback, verbose=verbose)
    df_face_info = df_face_info.dropna(axis=0)

    calc_sim = lambda ebd: (ebd_아나운서 * ebd).sum().item()
    df_face_info['similaraty'] = df_face_info['ebd'].map(calc_sim)
    df_face_info = df_face_info.sort_values(['frame_idx', 'similaraty'])
    
    # 유사도 기반으로 아나운서 얼굴만 필터(실제로는 먼저 가장 유사한 얼굴만 골라내기)
    return df_face_info, get_filtered_face(df_face_info, sim_th)


# save_face_info2 와 기능은 동일하나,
# 메모리 적게 사용하도록 개선한 버전
def save_face_info3(mp4_path, ebd_아나운서, base='./', callback=None, verbose=False):
    df_face_info_path = os.path.join(base,'df_face_info', f"{str(Path(mp4_path).stem)}.pickle")
    if verbose:
        print('save_face_info3 - df_face_info: ', str(df_face_info_path))
    callback1 = callback_inter(callback, min_per=0, max_per=90, desc='save_face_info3 - 1', verbose=verbose)
    callback2 = callback_inter(callback, min_per=90, max_per=100, desc='save_face_info3 - 2', verbose=verbose)
    
    fps = video_meta(mp4_path)['fps']
    stride = round(fps)*1
    if not Path(df_face_info_path).exists():
        r = get_face_info2(mp4_path, ebd_아나운서, stride=stride, callback=callback1, verbose=verbose)
        df_face_info, df_아나운서_only  = r
        os.makedirs(os.path.dirname(df_face_info_path), exist_ok=True)
        df_face_info.to_pickle(df_face_info_path)
        
    dst = Path(base) / 'df_anchor_i' / f"{Path(df_face_info_path).stem}_000.pickle"
    if verbose:
        print('df_anchor_i:', str(dst))
    if not Path(dst).exists():      
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        df = pd.read_pickle(df_face_info_path)
        df_ = df.sort_values('similaraty', ascending=False).drop_duplicates(['frame_idx'])
        df_ = df_.query('similaraty >= 0.3')
        #display(df_.groupby('frame_idx').count())
        #pdb.set_trace()
        df_face_info = face_info_to_anchor(df_, stride=stride, val_end=None)
        df_face_info.to_pickle(dst)
        return [dst]
    callback2(100)
    return [dst]

