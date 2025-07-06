import os
import socket
import torch.distributed as dist

is_distributed = True

def get_open_port() -> int:
    return _get_open_port()


def _get_open_port() -> int:
    port = 2333
    if port is not None:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                print(f"Port {port - 1} is already in use, trying port {port}")
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
        
        

def get_world_size():
    if is_distributed:
        return dist.get_world_size()
    else:
        return 1

def get_rank():
    if is_distributed:
        return dist.get_rank()
    else:
        return 0
    
def is_distributed():
    return is_distributed

def gather(tensor, all_tensors, rank=0):
    if is_distributed:
        dist.gather(tensor, all_tensors, rank)
    else:
        all_tensors[rank] = tensor

def all_reduce(tensor):
    if is_distributed:
        dist.all_reduce(tensor)