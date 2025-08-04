# ⚡ Layer-wise Energy Profiling of Deep Learning Models on Jetson Nano

This project performs **real-time, layer-wise power and energy profiling** of deep learning models (e.g., VGG11) during inference on **NVIDIA Jetson** devices Nano.
It uses `jtop` for live telemetry, isolates each layer, and exports detailed metrics and visualizations making it ideal for edge AI energy optimization.

---

## 🧠 Supported Deep Learning Models

* Any **PyTorch** model with a **sequential or decomposable structure**.
* Default:

  ```python
  torchvision.models.vgg11(pretrained=True)
  ```

---

## 🔧 Methodology

### 1. Layer Isolation

* Each layer (e.g., Conv2d, ReLU, Linear) is isolated and executed independently during inference.
* Input tensors are **pre-captured** using forward hooks during a dry run.
* This enables **accurate, per-layer energy attribution**.

### 2. Real-Time Power Sampling with `jtop`

* `jtop` provides access to **Power TOT** (total board power).
* A background thread samples power at a fixed `POLL_INTERVAL` (e.g., 0.02s).
* Only samples during the layer’s execution are recorded.

### 3. Repeated Inference for Stability

* Each layer is run `NUM_REPEATS` times (e.g., 100) to:

  * Smooth out noise
  * Ensure reproducibility
  * Capture stable power metrics

### 4. Energy & Power Calculation

For each layer:

* **Duration** = `end - start`
* **Energy (J)** =

  $$
  \sum (\text{Power sample} \times \text{Poll Interval})
  $$
* **Average Power (W)** =

  $$
  \text{Energy} / \text{Duration}
  $$

  (Physics-consistent: $E = P \times T$)

### 5. Data Logging & Visualization

* All raw samples are logged in:

  ```
  vgg11_layerwise_energy_full.csv
  ```
* A bar chart of **energy per layer** is saved as:

  ```
  vgg11_layerwise_energy_plot.png
  ```

---

## ✅ Why This Is Accurate and Useful

* 🟢 Real-time power telemetry from Jetson's onboard software interface (jtop)

* 🟢 Each layer is profiled in isolation during inference

* 🟢 Real input tensors from the model’s forward pass are used

* 🟢 Power usage is directly attributed to individual layers

* 🟢 Great for identifying inference-time bottlenecks and inefficiencies



---

## 🔢 Sample Output

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

## 📊 Live Power Monitoring with Prometheus & Grafana

This project also supports **real-time dashboards** via Prometheus + Grafana!

### 🔁 What It Does

* A Prometheus **exporter** reads power from `jtop` and exposes it at `http://<jetson-ip>:8000/metrics`
* **Prometheus** scrapes this data
* **Grafana** visualizes live power trends, inference stages, and energy usage

### 🛠 How To Set It Up

#### 1. Export Jetson Power

```bash
sudo pip install prometheus_client jetson-stats
python3 export_power.py
```

#### 2. Prometheus Config (`prometheus.yml`)

```yaml
scrape_configs:
  - job_name: 'jetson_power'
    static_configs:
      - targets: ['<jetson-ip>:8000']
```

Then:

```bash
./prometheus --config.file=prometheus.yml
```

#### 3. Grafana Dashboard

* Add Prometheus as a **data source**
* Create panels using:

  ```
  jetson_power_total_watts
  ```
* Plot energy as:

  ```
  increase(jetson_power_total_watts[60s]) * 60
  ```

> Optional files:

```
grafana_dashboard.json   # Prebuilt Grafana layout
export_power.py          # Exporter script
prometheus.yml           # Config
```

---


## 🔃 How to Run

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo python3 p.py
```

---

## 🧩 Requirements

* ✅ NVIDIA Jetson Nano / Xavier / TX2
* ✅ Python ≥ 3.8
* ✅ PyTorch + torchvision
* ✅ [jetson-stats](https://github.com/rbonghi/jetson-stats):

  ```bash
  sudo pip install jetson-stats
  ```

---

## 🧠 Applications

* 🔋 Optimize inference-time energy consumption for embedded deployment
* 🔎 Identify inefficient layers in real-time AI pipelines
* ⚙️ Evaluate runtime behavior of TensorRT vs PyTorch inference
* 📉 Benchmark low-power model variants (quantized, pruned, etc.)
* 🖥️ Visual dashboards with Grafana

---
