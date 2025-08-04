from prometheus_client import start_http_server, Gauge
from jtop import jtop
import time

power_metric = Gauge('jetson_power_total_watts', 'Total board power in watts')

def export_power():
    with jtop() as jetson:
        while jetson.ok():
            stats = jetson.stats
            power = stats.get("Power TOT", 0) / 1000.0  # in Watts
            power_metric.set(power)
            time.sleep(1)  # scrape interval

if __name__ == "__main__":
    start_http_server(8000)  # Prometheus scrapes from http://<ip>:8000/metrics
    export_power()
