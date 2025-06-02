import os
import torch
import numpy as np
import imageio
from pathlib import Path
from loguru import logger
from einops import rearrange
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from hymm_sp.config import parse_args
from hymm_sp.sample_inference_audio import HunyuanVideoSampler
from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal
from hymm_sp.data_kits.data_tools import save_videos_grid
from hymm_sp.data_kits.face_align import AlignImage
from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)

from transformers import WhisperModel
from transformers import AutoFeatureExtractor

MODEL_OUTPUT_PATH = os.environ.get('MODEL_BASE')


def main():
    args = parse_args()
    models_root_path = Path(args.ckpt)
    print("*"*20) 
    initialize_distributed(args.seed)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    print("+"*20)
    # Create save folder to save the samples
    save_path = args.save_path 
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    rank = 0
    vae_dtype = torch.float16
    device = torch.device("cuda")
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = torch.distributed.get_rank()

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    # Get the updated args
    args = hunyuan_video_sampler.args

    wav2vec = WhisperModel.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/").to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)
    
    BASE_DIR = f'{MODEL_OUTPUT_PATH}/ckpts/det_align/'
    det_path = os.path.join(BASE_DIR, 'detface.pt')    
    align_instance = AlignImage("cuda", det_path=det_path)
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/")

    kwargs = {
            "text_encoder": hunyuan_video_sampler.text_encoder, 
            "text_encoder_2": hunyuan_video_sampler.text_encoder_2, 
            "feature_extractor": feature_extractor, 
        }
    video_dataset = VideoAudioTextLoaderVal(
            image_size=args.image_size,
            meta_file=args.input, 
            **kwargs,
        )

    sampler = DistributedSampler(video_dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False)
    json_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, sampler=sampler, drop_last=False)

    for batch_index, batch in enumerate(json_loader, start=1):

        fps = batch["fps"]
        videoid = batch['videoid'][0]
        audio_path = str(batch["audio_path"][0])
        save_path = args.save_path 
        output_path = f"{save_path}/{videoid}.mp4"
        output_audio_path = f"{save_path}/{videoid}_audio.mp4"

        samples = hunyuan_video_sampler.predict(args, batch, wav2vec, feature_extractor, align_instance)
        
        sample = samples['samples'][0].unsqueeze(0)                    # denoised latent, (bs, 16, t//4, h//8, w//8)
        sample = sample[:, :, :batch["audio_len"][0]]
        
        video = rearrange(sample[0], "c f h w -> f h w c")
        video = (video * 255.).data.cpu().numpy().astype(np.uint8)  # （f h w c)
        
        torch.cuda.empty_cache()

        final_frames = []
        for frame in video:
            final_frames.append(frame)
        final_frames = np.stack(final_frames, axis=0)
        
        if rank == 0:
        #from hymm_sp.data_kits.ffmpeg_utils import save_video
        #save_video(final_frames, output_path, n_rows=len(final_frames), fps=fps.item())
        #imageio.mimsave(output_path, final_frames, fps=fps.item())
        #os.system(f"ffmpeg -i '{output_path}' -i '{audio_path}' -shortest '{output_audio_path}' -y -loglevel quiet; rm '{output_path}'")
        # 确保帧数据是 uint8 类型 (0-255)
        if final_frames.dtype != np.uint8:
            logger.warning(f"Frames are {final_frames.dtype}, attempting to convert to uint8.")
            if final_frames.max() <= 1.0 and final_frames.min() >= 0.0: # Heuristic for float 0-1
                final_frames = (final_frames * 255).astype(np.uint8)
            else: # Assume already in 0-255 range but wrong type
                final_frames = final_frames.astype(np.uint8)
    
        # 使用 imageio 输出高质量 MP4
        # ffmpeg_params 控制编码质量
        # -vcodec libx264: 使用 H.264 编码器
        # -crf 18: Constant Rate Factor, 0-51, 值越小质量越高，18-23 通常是很好的平衡
        # -preset medium: 编码速度与压缩率的平衡，可选: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        # -pix_fmt yuv420p: 像素格式，保证更广泛的播放器兼容性
        ffmpeg_video_params = [
            '-vcodec', 'libx264',
            '-crf', '16',
            '-preset', 'slower',
            '-pix_fmt', 'yuv420p'
        ]
        logger.info(f"Saving intermediate video to: {output_path} with fps: {fps.item()}")
        imageio.mimsave(output_path, final_frames, format='mp4', fps=fps.item(), ffmpeg_params=ffmpeg_video_params)
    
        # ffmpeg 命令合并音视频
        # -c:v copy: 直接复制视频流，不重新编码，前提是 imageio 输出的视频已经是我们想要的格式和质量
        # -c:a aac: 使用 AAC 音频编码 (如果原始音频不是 AAC 或者需要控制比特率)
        # -b:a 192k: 音频比特率 (可选)
        # -shortest: 以最短的输入流（视频或音频）长度为准
        # -y: 覆盖输出文件不提示
        # -loglevel quiet: 安静模式
        ffmpeg_merge_cmd = (
            f"ffmpeg -i '{output_path}' -i '{audio_path}' "
            f"-c:v copy -c:a aac -shortest '{output_audio_path}' -y -loglevel quiet"
        )
        logger.info(f"Merging video and audio. Command: {ffmpeg_merge_cmd}")
        result_code = os.system(ffmpeg_merge_cmd)
    
        if result_code == 0:
            logger.info(f"Successfully created video with audio: {output_audio_path}")
            logger.info(f"Removing intermediate video: {output_path}")
            os.remove(output_path)
        else:
            logger.error(f"ffmpeg command failed with exit code {result_code}. Intermediate file '{output_path}' not removed.")




    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
