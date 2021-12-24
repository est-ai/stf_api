from .preprocess_dir.utils import face_finder as ff
from .preprocess_dir.utils import crop_with_fan as cwf 
from pathlib import Path
import imageio
import sys
import gc
from .util import *
from .gen_video import gen_video

    
# template video 전처리
def preprocess_template(config_path, template_video_path, reference_face, work_root_path, call_back, device, verbose=False):  
    ff.init_face_finder(device)
    cwf.init_fan(device)
    config = read_config(config_path)
    
    preprocess_dir = get_preprocess_dir(work_root_path, config.name)
    Path(preprocess_dir).mkdir(exist_ok=True, parents=True)
    # snow : for debug
    if verbose:
        print('preprocess_dir: ', preprocess_dir, ', work_root_path:', work_root_path)
    
    # 전처리 파일 경로
    crop_mp4 = get_crop_mp4_dir(preprocess_dir, template_video_path)

    if not Path(crop_mp4).exists():
        if verbose:
            print('템플릿 비디오 처리 ... ')
        #아나운서 얼굴 정보를 구한다.
        df_face, imgs = ff.find_face(reference_face)
        g_anchor_ebd = df_face['ebd'].values[0]

        # 템플릿 동영상에서 아나운서 얼굴 위치만 저장해 놓는다
        df_paths = ff.save_face_info2(template_video_path, g_anchor_ebd, base=preprocess_dir, verbose=verbose)

        ### 얼굴 영역을 FAN 랜드마크 기반으로 크롭해 놓는다
        assert len(df_paths) == 1
        if verbose:
            print('cwf.save_crop_info --')
        df_fan_path = cwf.save_crop_info(anchor_box_path=df_paths[0],
                                         mp4_path=template_video_path,
                                         out_dir=crop_mp4,
                                         crop_offset_y = config.crop_offset_y,
                                         crop_margin = config.crop_margin,
                                         verbose=verbose,
                                         )
        # snow : for debug
        if verbose:
            print('df_fan_path: ', df_fan_path)
    else:
        if verbose:
            print('전처리가 이미 되어있음')

    gc.collect()


class TemplateVideo():
    def __init__(self, model, template_video_path, frames, verbose=False):
        self.model = model
        self.template_video_path = template_video_path
        self.full_frames = frames 
        self.verbose = verbose
        self.preprocess_dir = get_preprocess_dir(model.work_root_path, model.args.name)
        self.crop_mp4_dir = get_crop_mp4_dir(self.preprocess_dir, template_video_path)
        self.fps = ff.video_meta(template_video_path)['fps']


    def __del__(self):
        if self.verbose:
            print('del model , gc:',  sys.getrefcount(self.full_frames))
        del self.full_frames
        gc.collect()
        if self.verbose:
            print('del template, gc:', gc.get_count())


    def gen(self, wav_path, wav_std, wav_std_ref_wav, video_start_offset_frame, slow_write=True, out_path=None):
        if out_path is None:
            out_path = 'temp.mp4'
        return gen_video(self, wav_path, wav_std, wav_std_ref_wav,
                         video_start_offset_frame, out_path,
                         slow_write=slow_write, verbose=self.verbose)
        
        
def template_video(model, template_video_path, callback, verbose=False):
    if verbose:
        print('load template video : ', template_video_path)
    frames = imageio.mimread(template_video_path, 'ffmpeg', memtest=False)
    template_video = TemplateVideo(model, template_video_path, frames, verbose=verbose)
    if verbose:
        print('complete loading template video : ', template_video_path)
    return template_video

