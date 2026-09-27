"""Minimal torchrun helpers shared by train_qlora.py / train_rft.py / train_grpo.py.

Data-parallel without the DDP wrapper: every rank holds a full (quantized) model copy, and the
trainable LoRA gradients are averaged with one flat all-reduce right before the optimizer
step. That keeps gradient accumulation, several forwards per backward (GRPO confidence),
`disable_adapter()` reference passes and non-reentrant gradient checkpointing all legal,
none of which DDP's per-parameter hooks tolerate. LoRA gradients are a few tens of MB, so
one all-reduce per optimizer step is negligible over NVLink.

Single-process runs (plain `python train_x.py`) work unchanged: world size 1, no collectives.
"""

import datetime
import os

import torch
import torch.distributed as dist


def init():
    """Returns (rank, world_size, local_rank) and pins this process to its GPU."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        # Long timeout: rank 0 may download the model or save while the others wait.
        minutes = int(os.environ.get("JEV_NCCL_TIMEOUT_MIN", "60"))
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=minutes),
                                device_id=torch.device("cuda", local_rank))
    return rank, world, local_rank


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def is_main():
    return rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def print0(*args, **kwargs):
    if is_main():
        print(*args, flush=True, **kwargs)


def shard(seq):
    """Round-robin slice of seq for this rank (uneven lengths are fine for gather-based eval)."""
    return seq[rank()::world_size()]


def gather(obj):
    """all_gather_object: list with one entry per rank, on every rank."""
    if not dist.is_initialized():
        return [obj]
    out = [None] * world_size()
    dist.all_gather_object(out, obj)
    return out


def all_reduce_sum(values):
    """Sum a dict of floats across ranks (same keys on every rank)."""
    if not dist.is_initialized():
        return dict(values)
    keys = sorted(values)
    t = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device="cuda")
    dist.all_reduce(t)
    return dict(zip(keys, t.tolist()))


def any_true(flag):
    """True on every rank if flag is True on any rank (keeps early stops in lockstep)."""
    if not dist.is_initialized():
        return bool(flag)
    t = torch.tensor([1.0 if flag else 0.0], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


@torch.no_grad()
def broadcast_params(params):
    """Make rank 0's (randomly initialised) LoRA weights the starting point everywhere."""
    if not dist.is_initialized():
        return
    for p in params:
        dist.broadcast(p.data, src=0)


@torch.no_grad()
def sync_grads(params):
    """Average .grad over ranks with one flat all-reduce. Missing grads count as zero."""
    if not dist.is_initialized():
        return
    params = list(params)
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    offset = 0
    for p, g in zip(params, grads):
        n = g.numel()
        chunk = flat[offset:offset + n].view_as(g).to(g.dtype)
        if p.grad is None:
            p.grad = chunk
        else:
            p.grad.copy_(chunk)
        offset += n


def fetch_model(model_id, revision):
    """Download once on rank 0 so eight ranks do not race on the same HF cache files."""
    if is_main():
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, revision=revision)
    barrier()
