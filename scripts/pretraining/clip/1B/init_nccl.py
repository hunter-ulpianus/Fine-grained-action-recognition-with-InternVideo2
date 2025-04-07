import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

def setup(rank, world_size):
    """初始化分布式进程组"""
    dist.init_process_group(
        backend="nccl",  # NCCL 是 GPU 上的高性能通信库
        init_method="env://",  # 通过环境变量进行初始化
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)  # 每个进程绑定到一个 GPU
    print(f"Rank {rank}/{world_size} process initialized.")

def cleanup():
    """销毁分布式进程组"""
    dist.destroy_process_group()
    print("Process group destroyed.")

def train(rank, world_size):
    """训练逻辑"""
    setup(rank, world_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, required=True, help="Total number of processes")
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(setup, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

