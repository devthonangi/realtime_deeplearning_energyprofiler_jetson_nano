# Layer-wise Energy Profiling of Deep Learning Models on Jetson Nano

This project performs **layer-wise energy and power profiling** of deep learning models (e.g., VGG11) during **inference** on NVIDIA Jetson devices. It uses `jtop` to monitor live power telemetry and attributes energy consumption to individual layers.

---

##  Supported Deep Learning Model

This setup supports any PyTorch model with a sequential or decomposable structure. By default, it uses:

```python
torchvision.models.vgg11(pretrained=True)
```

---

##  Methodology

###  1. Layer Isolation

* Each layer (e.g., `Conv2d`, `ReLU`, `Linear`) is isolated and executed **independently**.
* Input tensors for each layer are captured using **forward hooks** during a dry run.
* This ensures that only one layer runs during profiling, and its energy usage is measured accurately.

---

###  2. Real-Time Power Sampling with `jtop`

* `jtop` is used to read `Power TOT`, which represents total board power usage.
* A **background thread** samples power values at a fixed `POLL_INTERVAL` (e.g., 0.02s).
* Only the power readings captured **during layer execution** are considered.

---

###  3. Repeated Inference for Stability

* Each layer is executed **`NUM_REPEATS` times (e.g., 100)**.
* This reduces measurement noise and smooths out transient hardware fluctuations.
* Ensures more stable and representative power data per layer.

---

###  4. Energy and Power Calculation

* Duration is calculated using `time.monotonic()` before and after the loop.

* **Energy (Joules)** is computed as:

  ```
  Energy = sum(Power samples) × Poll Interval
  ```

* **Average Power (Watt)** is calculated as:

  ```
  Avg Power = Energy / Duration
  ```

* This is consistent with the physics principle:
  `E = P × T`

---

###  5. Data Logging and Visualization

* All **raw power samples per layer** are stored in:
  `vgg11_layerwise_energy_full.csv`

* A **bar chart** is generated showing:

  * Energy consumed per layer (J)
  * Chart saved as:
    `vgg11_layerwise_energy_plot.png`

---

##  Why This is Accurate and Useful

* Power telemetry is accessed **in real time** using `jtop` (via Jetson's internal software interface).
* Each layer is run **in isolation with real input tensors**.
* Power usage is linked directly to that specific layer.
* Helps identify **energy hotspots** in the model and optimize performance for **edge/embedded deployment**.

---

## Sample Output

```
===== Layer-wise Profiling (VGG11) =====
Conv2d_0: Duration=2.45s | Energy=18.51J | Avg Power=7.57W
ReLU_1:   Duration=0.71s | Energy=3.65J  | Avg Power=5.16W
...

===== Total Summary =====
Total Duration: 38.2 s
Total Energy:   227.2 J
Avg Total Power: 5.93 W
```

---

## Requirements

* NVIDIA Jetson device (Nano, Xavier, etc.)
* Python 3.8+
* PyTorch, torchvision
* `jtop` (install with: `sudo pip install jetson-stats`)

---

##  How to Run

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo python3 p.py
```

---
