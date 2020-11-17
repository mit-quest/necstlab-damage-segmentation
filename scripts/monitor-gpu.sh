## allow monitoring gpu performance
if [ -d "/monitor-gpu" ]
then
    cd monitor-gpu/gcp-gpu-utilization-metrics
    python3 ./report_gpu_metrics.py &
    echo "GPU monitoring ON"
else
    mkdir -p monitor-gpu
    cd monitor-gpu
    git clone https://github.com/b0noI/gcp-gpu-utilization-metrics.git
    cd gcp-gpu-utilization-metrics && pip3 install -r requirenments.txt
    python3 ./report_gpu_metrics.py &
    echo "GPU monitoring ON"
fi