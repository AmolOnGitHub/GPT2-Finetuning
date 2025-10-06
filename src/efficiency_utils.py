import os, time, math, torch, numpy as np
from torch.utils.data import DataLoader


def get_device(device_arg: str) -> str:
    """
    Determine the computation device based on user input and availability.
    """
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_synchronize(device: str):
    """
    Synchronize the specified device to ensure accurate timing measurements.
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        try: torch.mps.synchronize()
        except Exception: pass


def folder_size_bytes(path: str) -> int:
    """
    Calculate the total size of files in the specified folder (in bytes).
    """
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def pretty_size(num_bytes: int) -> str:
    """
    Convert a size in bytes to a human-readable string format.
    """
    if num_bytes < 1024: return f"{num_bytes} B"
    kb = num_bytes / 1024
    if kb < 1024: return f"{kb:.2f} KB"
    mb = kb / 1024
    if mb < 1024: return f"{mb:.2f} MB"
    gb = mb / 1024
    return f"{gb:.2f} GB"


def measure_latency(model, dataset, data_collator, device: str, batch_size=1, warmup_steps=20, measure_steps=100):
    """
    Measure the average inference latency (in ms per sample) of the model on the given dataset.
    """

    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    times, steps = [], 0
    
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            t0 = time.perf_counter()
            _ = model(**batch)
            device_synchronize(device)
            t1 = time.perf_counter()
            if steps >= warmup_steps:
                times.append(t1 - t0)
            steps += 1
            if steps >= warmup_steps + measure_steps:
                break

    if not times: return float("nan")
    avg_batch_ms = np.mean(times) * 1000.0
    return avg_batch_ms / batch_size
