import stf
from pathlib import Path
from glob import glob
import argparse
import os
import time


def main(args):
    s1 = time.time()

    stf.preprocess_template(config_path=args.config_path,
                            template_video_path=args.template_video_path,
                            reference_face=args.reference_face,
                            work_root_path=args.work_root_path,
                            callback=None,
                            device=args.device,
                            verbose=args.verbose)
    print('1 preprocess_template ')

    model = stf.create_model(config_path=args.config_path,
                             checkpoint_path=args.checkpoint_path,
                             work_root_path=args.work_root_path,
                             device=args.device,
                             verbose=args.verbose)
    print('2 create_model ')

    template = stf.template_video(model=model,
                                  template_video_path=args.template_video_path,
                                  callback=None,
                                  verbose=args.verbose)
    print('3 template_video ')
    s2 = time.time()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    template.gen4(wav_path=args.wav_path,
                 wav_std=True,
                 wav_std_ref_wav=args.wav_std_ref_wav,
                 video_start_offset_frame=args.video_start_offset_frame, 
                 slow_write=False,
                 head_only=False,
                 out_path=args.output,
                 )
    print('#### ', time.time() - s1)
    print('#### ', time.time() - s2)
    print('4 gen ')
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='output root')
    parser.add_argument('--wav_path', type=str, help='wav_path')
    parser.add_argument('--template_video_path', type=str, help='template_video_path')
    parser.add_argument('--config_path',  type=str, help='config_path path')
    parser.add_argument('--checkpoint_path',  type=str, help='checkpoint_path path')
    parser.add_argument('--work_root_path',  type=str, help='work_root_path')
    parser.add_argument('--wav_std_ref_wav',  type=str, help='wav_std_ref_wav path')
    parser.add_argument('--reference_face',  type=str, help='face image path')
    parser.add_argument('--device', type=str, help='like "cuda:1" ')
    parser.add_argument('--video_start_offset_frame', type=int, help='video_start_offset_frame ')
    
    parser.add_argument('--verbose',  action='store_true')
    args = parser.parse_args()
    print(args)

    # parameter 확인
    if not os.path.exists(args.wav_path):
        print('file is not exist:', args.wav_path)
        exit()
    if not os.path.exists(args.template_video_path):
        print('file is not exist:', args.template_video_path)
        exit()
    if not os.path.exists(args.config_path):
        print('file is not exist:', args.config_path)
        exit()
    if not os.path.exists(args.checkpoint_path):
        print('file is not exist:', args.checkpoint_path)
        exit()
    if not os.path.exists(args.wav_std_ref_wav):
        print('file is not exist:', args.wav_std_ref_wav)
        exit()
    if not os.path.exists(args.reference_face):
        print('file is not exist:', args.reference_face)
        exit()

    main(args)
