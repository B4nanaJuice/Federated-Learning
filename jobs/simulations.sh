# Clean simulation
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 0 \
    --client-fraction 0.5 \
    --epochs 15 \
    --save-filename "clean_run"

# 5% malicious clients attacking their data continuously
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 1 \
    --client-fraction 0.5 \
    --epochs 15 \
    --save-filename "5%_clients_data" \
    --client-attack-rate "lambda x: True" \
    --client-attack-target data

# 20% malicious clients attacking their data continuously
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 4 \
    --client-fraction 0.5 \
    --epochs 15 \
    --save-filename "20%_clients_data" \
    --client-attack-rate "lambda x: True" \
    --client-attack-target data

# 5% malicious clients attacking the model continuously
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 1 \
    --client-fraction 0.5 \
    --epochs 15 \
    --save-filename "5%_clients_model" \
    --client-attack-rate "lambda x: True"

# 20% malicious clients attacking the model continuously
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 4 \
    --client-fraction 0.5 \
    --epochs 15 \
    --save-filename "20%_clients_model" \
    --client-attack-rate "lambda x: True"

# 20% malicious clients attacking the model continuously with gradient inversion
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 4 \
    --client-fraction 0.5 \
    --epochs 15 \
    --save-filename "clients_gradient_inversion" \
    --client-attack-rate "lambda x: True" \
    --client-attack-method "gradient-inversion"

# 20% malicious clients attacking the model continuously with gradient amplification
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 4 \
    --client-fraction 0.5 \
    --epochs 15 \
    --save-filename "clients_gradient_amplification" \
    --client-attack-rate "lambda x: True" \
    --client-attack-method "gradient-amplification"

# Attacker that will poison only last layer of the broadcasted model
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 0 \
    --client-fraction 0.5 \
    --epochs 15 \
    --attacked-server True \
    --save-filename "partial_corruption" \
    --server-attack-rate "lambda x: x in [7, 8, 12, 14]" \
    --partial_attack True

# Attacker that has total access on server's model
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 0 \
    --client-fraction 0.5 \
    --epochs 15 \
    --attacked-server True \
    --save-filename "total_takeover" \
    --server-attack-rate "lambda x: x == 8"

# Server has scoring system with distance metric and test only with honnest clients
python run.py run-simulation --run-count 10 \
    --save-filename distance_scoring \
    --rounds 15 \
    --server-scoring true \
    --metric distance \
    --threshold 0.4 \
    --sigma 1,2,3,4,4.5,5,5.5,6,8,10 \
    --client-count 5 \
    --epochs 15 \
    --batch 128 \
    --fraction 1