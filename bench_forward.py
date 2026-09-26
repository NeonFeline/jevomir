"""Measure forward throughput on in-memory items (no download time)."""
import sys, time, torch
from cauldron_tasks import iter_items
from extract_cauldron import forward_batch, letter_ids, load_model
from extract_onepass import auto_layers

items = list(iter_items("clevr", 128)) + list(iter_items("vqav2", 128))
model, processor = load_model()
layers, ids = auto_layers(model), letter_ids(processor.tokenizer)
forward_batch(model, processor, items[:8], layers, ids)  # warmup
for bs in (16, 32, 64):
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(0, len(items), bs):
        forward_batch(model, processor, items[i:i + bs], layers, ids)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    print(f"batch {bs}: {len(items) / dt:.1f} items/s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
