## allow monitoring gpu performance
## scripts in monitor_gpu folder taken from https://github.com/b0noI/gcp-gpu-utilization-metrics
## fixed requirements (not fixed in the git, and that leads to errors)
cd scripts/monitor_gpu && pip3 install -r requirenments.txt
python3 ./report_gpu_metrics.py &
echo "GPU monitoring ON"
