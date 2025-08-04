import torch
import torch.nn as nn
import torchvision.models as models
import time
from jtop import jtop
import threading
import queue
import matplotlib.pyplot as plt
import csv

# --- Configuration ---
NUM_REPEATS = 100
POLL_INTERVAL = 0.02
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg11(weights=models.VGG11_Weights.DEFAULT).to(DEVICE).eval()
layers = list(model.features) + [model.avgpool] + list(model.classifier)

# --- Power Logger Thread ---
def power_logger(stop_event, power_q):
    with jtop() as jetson:
        while not stop_event.is_set():
            stats = jetson.stats
            power = stats.get("Power TOT", 0) / 1000.0  # mW to W
            ts = time.monotonic()
            power_q.put((ts, power))
            time.sleep(POLL_INTERVAL)

# --- Run a single layer with power measurement ---
def run_layer(layer, input_tensor):
    power_q = queue.Queue()
    stop_event = threading.Event()
    power_thread = threading.Thread(target=power_logger, args=(stop_event, power_q))
    power_thread.start()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = layer(input_tensor)

    start = time.monotonic()
    with torch.no_grad():
        for _ in range(NUM_REPEATS):
            _ = layer(input_tensor)
    end = time.monotonic()

    stop_event.set()
    power_thread.join()

    duration = end - start
    samples = []
    while not power_q.empty():
        t, p = power_q.get()
        if start <= t <= end:
            samples.append(p)
    energy = sum(samples) * POLL_INTERVAL
    avg_power = energy / duration if duration > 0 else 0

    return duration, energy, avg_power, samples

# --- Hook to get layer inputs ---
layer_inputs = {}

def save_input(module, input, output):
    layer_inputs[module] = input[0].detach()

hooks = []
x = torch.randn(1, 3, 224, 224).to(DEVICE)
for layer in layers:
    hooks.append(layer.register_forward_hook(save_input))
with torch.no_grad():
    _ = model(x)
for h in hooks:
    h.remove()

# --- Layer Profiling ---
print("\n===== Layer-wise Profiling (VGG11) =====")
results = []
total_energy = 0
total_duration = 0

for i, layer in enumerate(layers):
    name = f"{layer.__class__.__name__}_{i}"
    input_tensor = layer_inputs.get(layer, None)
    if input_tensor is None:
        print(f"[SKIP] {name}: No input captured")
        continue
    try:
        duration, energy, avg_power, samples = run_layer(layer, input_tensor.to(DEVICE))
        total_energy += energy
        total_duration += duration
        results.append((name, duration, energy, avg_power, samples))
        print(f"{name}: Duration={duration:.3f}s | Energy={energy:.3f}J | Avg Power={avg_power:.2f}W")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")

# --- CSV Export with Power Samples ---
csv_path = "vgg11_layerwise_energy_full.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Layer", "Duration (s)", "Energy (J)", "Avg Power (W)", "Power Samples (W)"])
    for name, duration, energy, avg_power, samples in results:
        sample_str = ",".join(f"{s:.2f}" for s in samples)
        writer.writerow([name, f"{duration:.3f}", f"{energy:.3f}", f"{avg_power:.2f}", sample_str])

print(f"\n Raw power data exported to: {csv_path}")

# --- Plot Summary (optional visualization) ---
plt.figure(figsize=(12, 6))
layer_names = [r[0] for r in results]
energies = [r[2] for r in results]
durations = [r[1] for r in results]

plt.bar(layer_names, energies)
plt.xticks(rotation=90)
plt.ylabel("Energy (Joules)")
plt.title("VGG11 Layer-wise Energy Consumption")
plt.tight_layout()
plt.savefig("vgg11_layerwise_energy_plot.png")
print(" Plot saved to: vgg11_layerwise_energy_plot.png")

# --- Final Summary ---
print("\n===== Total Summary =====")
print(f"Total Duration: {total_duration:.3f} s")
print(f"Total Energy:   {total_energy:.3f} J")
print(f"Avg Total Power: {total_energy / total_duration:.2f} W" if total_duration > 0 else "N/A")
