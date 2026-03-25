

Stop-End-Messages Experiment

# Stop-End-Messages Experiment

## Purpose
This experiment studies how stopping real-message generation near the end of the simulation affects anonymity and traffic behavior in a stratified mix network.

## Experiment Sweep
- Number of runs: 1
- Number of layers: 3
- Mixes per layer tested: [3, 4, 5, 6, 7, 8, 9, 10]
- Stop-real-message percentages tested: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]

## Runtime Settings
- Burnout: 0
- Simulation duration: 20

## Base Configuration
The following configuration is used as the baseline template:

[DEFAULT]
n_clients = 100
lambda_c = 1

[TOPOLOGY]
# Topology options: stratified, cascade, free route
type = stratified
fully_connected = True
# Routing options: source, hopbyhop
routing = source
E2E = 2
n_layers = 3
# Used when type = stratified
l_mixes_per_layer = 3
# Used when type = cascade
n_cascades = 3

[MIXING]
# mix_type options: poisson, time, pool
mix_type = poisson
mu = 1
timeout = 2
threshold = 100
flush_percent = 0.1
# Stop sending real messages during the last X% of simulation time
stop_real_msgs_percent = 10

[DUMMIES]
client_dummies = False
rate_client_dummies = 1
link_based_dummies = True
multiple_hop_dummies = False
rate_mix_dummies = 1

[NODES_SELETION]
# Node-selection probability mode: Uniform or specific
probability = Uniform
# Used when probability = specific and type = stratified
w_mix_l1 = [0.5]
w_mix_l2 = [0.3]
w_mix_l3 = [0.1]
# Used when probability = specific and type = cascade
w_cascades = [0.8, 0.1, 0.1]

[THREATMODEL]
corrupt_mixes = 0
balanced_corruption = False
