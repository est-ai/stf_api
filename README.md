# stf_api
SW사업본부 제공용 stf api


# 설치
CUDA 11.1 기준  설치방법

```
$ sudo apt-get install ffmpeg
$ conda create --name stf python==3.7.9
$ conda activate stf
$ git clone https://github.com/ai-anchor-kr/stf_api.git
$ cd stf_api
$ conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
$ pip install -r requirements.txt

```

# Release
* 경로 : https://drive.google.com/drive/folders/1abifSj7SO2v_zMBu98EWDJ8c4UPkJ2-M
* versions
    * 220105.zip
      * 테스트/검토용 wav/mp4 파일과 서비스에서 필요한 파일 구분을 위해 폴더 구조를 변경
      * model weight 에서 불필요한 부분 제거하여 용량을 감소시킴.
      * 폴더 내 파일 설명    
    ```
    220105/
        release_220105/ : 서비스 필요한 파일들
            front_1213_1140_105.pth : 정면 model weight
            front_config.json : 정면 model 설정파일
            side_1214_1245_039.pth : 측면 model weight
            side_config.json : 측면 model 설정파일
            hunet.png : stf api 제공할 이민영 이미지.
            
        demo_src/ : 테스트용 wav, template 파일들
            wav_211216/ : 테스트용 wav 들이 있는 directory
            Est Soft 2_1.0-80.mov : 테스트용 템플릿(정면 비디오)
            Est Soft 2_2.0-80.mov : 테스트용 템플릿(측면 비디오)
            hunet_only_voice_test_50000.wav : 테스트용 wav
            
        result_sample/ : 검토용 stf 생성 비디오
            front-with_video-250000_001.wav_s_lr0001_ep105_Est Soft 2_1.0-80_2차_jh_hunet_with_video_filtered_250000_001_mask-pwb_front_v39_1_VSO-0_352px_105.pth.mp4  : 정면 stf 비디오
            side-with_video-250000_001.wav_s_lr0001_ep39_Est Soft 2_2.0-80_2차_jh_hunet_with_video_filtered_250000_001_mask-pwb_side_v39_12_VSO-0_352px_039.pth.mp4   : 측면 stf 비디오
    
    ```
    * 211230.zip
      * 최초 release
      * 폴더 내 파일 설명
    
    ```
    211230/
        front_1213_1140_105.pth : 정면 model weight
        front_config.json : 정면 model 설정파일
        side_1214_1245_039.pth : 측면 model weight
        side_config.json : 측면 model 설정파일
        hunet_only_voice_test_50000.wav : 테스트용 wav
        hunet.png : stf api 제공할 이민영 이미지.
        
        demo_src/ : 테스트용 wav, template 파일들
            wav_211216/ : 테스트용 wav 들이 있는 directory
            Est Soft 2_1.0-80.mov : 테스트용 템플릿(정면 비디오)
            Est Soft 2_2.0-80.mov : 테스트용 템플릿(측면 비디오)
            
        result_sample/ : 검토용 stf 생성 비디오
            front-with_video-250000_001.wav_s_lr0001_ep105_Est Soft 2_1.0-80_2차_jh_hunet_with_video_filtered_250000_001_mask-pwb_front_v39_1_VSO-0_352px_105.pth.mp4  : 정면 stf 비디오
            side-with_video-250000_001.wav_s_lr0001_ep39_Est Soft 2_2.0-80_2차_jh_hunet_with_video_filtered_250000_001_mask-pwb_side_v39_12_VSO-0_352px_039.pth.mp4   : 측면 stf 비디오
    
    ```
