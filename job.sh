#! /bin/bash
# Example of job file
./venv/Scripts/python.exe run.py run-simulation \
    --run-count 5 \
    --max-rounds 3 \
    --total-clients 5 \
    --malicious-client-count 0 \
    --client-fraction 0.4 \
    --epochs 2 \
    --min-clients 0