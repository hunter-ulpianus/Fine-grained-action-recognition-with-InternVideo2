import decord
from decord import VideoReader, gpu
import random
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing
import subprocess
from PIL import Image

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_P2P_DISATLE'] = '1'

video_path = "/media/sdc/datasets/ssv2/train_data/"
save_path = "/media/sdc/datasets/ssv2/train_fms/"

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, trimmed30=False, gpu_id=None
    ):
    num_threads = 1 if video_path.endswith('.webm') else 0 # make ssv2 happy
    video_reader = VideoReader(video_path, num_threads=num_threads)
    vlen = len(video_reader)
 
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    # only use top 30 seconds
    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )

    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, duration

def save_frames_as_images(frames, save_dir, video_name, frame_indices):
    video_save_dir = os.path.join(save_dir, video_name)
    os.makedirs(video_save_dir, exist_ok=True)

    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame.astype('uint8'))

        width, height = img.size
        left = (width - 224) / 2
        top = (height - 224) / 2
        right = (width + 224) / 2
        bottom = (height + 224) / 2
        img_cropped = img.crop((left, top, right, bottom))

        img_save_path = os.path.join(video_save_dir, f"frame_{i:02d}.png")
        img_cropped.save(img_save_path)
    # print(f"Saved {len(frames)} frames to {video_save_dir}")

def process_videos(rank, world_size, video_list, args):

    device = torch.device("cuda", rank)
    print(f"Rank {rank} starting...Using device {device}")
    total_videos = len(video_list)
    chunk = total_videos // world_size
    start = rank * chunk
    end = total_videos if rank == world_size -1 else (rank + 1) * chunk
    sub_video_list = video_list[start:end]

    for idx, video_name in enumerate(sub_video_list):
        video_full_path = os.path.join(args["video_path"], video_name)
        video_name_no_ext = os.path.splitext(video_name)[0]


        frames, frame_indices, duration = read_frames_decord(
            video_full_path, 
            num_frames=8,
            sample='rand',
            fix_start=None,
            max_num_frames=-1,
            trimmed30=False,
            gpu_id=rank
        )

        save_frames_as_images(
            frames,
            save_dir=args["save_path"],
            video_name=video_name_no_ext,
            frame_indices=frame_indices
        )

        if (idx + 1) % 10 == 0:
            print(f"[GPU{rank}] Processed {idx+1} / {len(sub_video_list)} videos")

    dist.barrier()
    dist.destroy_process_group()

def main():
    args = {
        "video_path": video_path,
        "save_path": save_path
    }
    os.makedirs(args["save_path"], exist_ok=True)

    video_list = os.listdir(args["video_path"])
    video_list = [v for v in video_list if v.lower().endswith(('.webm'))]
    
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs, starting distributed run...")

    with multiprocessing.get_context('spawn').Pool(processes=world_size) as executor:
        executor.starmap(process_videos, [(rank, world_size, video_list, args) for rank in range(world_size)])
    print("All GPUs have finished processing")
    return 

if __name__ == "__main__":
    main()