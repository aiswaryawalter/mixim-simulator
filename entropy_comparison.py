import pandas as pd
import matplotlib.pyplot as plt
import os
import configparser
from datetime import datetime
from Simulation import Simulation
from util import Weights
import numpy as np
import pickle
import json

def run_simulation(n_layers, n_mixes_per_layer, config):
    """Run a single simulation and return entropy array for all targets"""
    topology_vars = config['TOPOLOGY']
    topology = topology_vars['type']
    fully_connected = topology_vars.getboolean('fully_connected')
    routing = topology_vars['routing']
    mix_type = config['MIXING']['mix_type']

    n_clients = int(config['DEFAULT']['n_clients'])
    lambda_c = float(config['DEFAULT']['lambda_c'])
    n_cascade = int(config['TOPOLOGY']['n_cascades'])

    mu = (int(config['TOPOLOGY']['E2E']) - (n_layers + 1)*0.05) / n_layers
    threshold = int(config['MIXING']['threshold'])
    flush_percent = float(config['MIXING']['flush_percent'])
    timeout = float(config['MIXING']['timeout'])

    corrupt_mixes = int(config['THREATMODEL']['corrupt_mixes'])
    balanced_corruption = bool(config['THREATMODEL']['balanced_corruption'])

    dummies_vars = config['DUMMIES']
    client_dummies = dummies_vars.getboolean('client_dummies')
    rate_client_dummies = float(dummies_vars['rate_client_dummies'])
    link_dummies = dummies_vars.getboolean('link_based_dummies')
    multiple_hops_dummies = dummies_vars.getboolean('multiple_hop_dummies')
    rate_mix_dummies = float(dummies_vars['rate_mix_dummies'])

    weights = Weights(n_layers, n_mixes_per_layer)
    simulation = Simulation(
        mix_type=mix_type, simDuration=50, rate_client=1/lambda_c, mu=mu, logging=True,
        topology=topology, fully_connected=fully_connected, n_clients=n_clients, 
        flush_percent=flush_percent, printing=False, flush_timeout=timeout, 
        threshold=threshold, routing=routing, n_layers=n_layers,
        n_mixes_per_layer=n_mixes_per_layer, corrupt=corrupt_mixes, 
        unifrom_corruption=balanced_corruption, probability_dist_mixes=weights,
        nbr_cascacdes=n_cascade, client_dummies=client_dummies,
        rate_client_dummies=rate_client_dummies, link_based_dummies=link_dummies, 
        multiple_hops_dummies=multiple_hops_dummies, rate_mix_dummies=rate_mix_dummies,
        Network_template=None
    )

    entropy, entropy_mean, entropy_median, entropy_q25 = simulation.run()
    return entropy  # Return full entropy array for all targets

def main():
    # Configuration
    config = configparser.ConfigParser()
    config.read('ConfigFile.ini')
    
    logDir = 'Logs/'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = f'{logDir}entropy_comparison_{timestamp}/'
    os.makedirs(plots_dir, exist_ok=True)

    # Parameters
    layers_list = [3, 4, 5]
    mixes_list = [3, 4, 5, 6, 7, 8, 9, 10]
    n_runs = 15

    # Store results: {layers: {mixes: [entropy_arrays_from_runs]}}
    results = {}
    
    print("Starting entropy comparison analysis...")
    print(f"Layers: {layers_list}")
    print(f"Mixes per layer: {mixes_list}")
    print(f"Runs per combination: {n_runs}\n")

    # Run simulations
    for n_layers in layers_list:
        results[n_layers] = {}
        for n_mixes in mixes_list:
            print(f"Running: {n_layers} layers, {n_mixes} mixes per layer...")
            
            entropy_arrays = []  # Store entropy arrays from all runs
            
            for run in range(n_runs):
                print(f"  Run {run + 1}/{n_runs}", end='\r')
                try:
                    entropy_array = run_simulation(n_layers, n_mixes, config)
                    entropy_arrays.append(entropy_array)
                except Exception as e:
                    print(f"  Error in run {run + 1}: {e}")
                    continue
            
            print(f"  Run {n_runs}/{n_runs} - Complete     ")
            
            # Store all entropy arrays for this combination
            results[n_layers][n_mixes] = entropy_arrays
            print(f"  Collected {len(entropy_arrays)} runs\n")

    # Save raw results to pickle file
    results_pickle = f'{logDir}{timestamp}results_raw.pkl'
    with open(results_pickle, 'wb') as f:
        pickle.dump(results, f)
    print(f"Raw results saved to {results_pickle}")

        # Save processed results to CSV for easy access
    detailed_data = []
    for n_layers in layers_list:
        for n_mixes in mixes_list:
            if n_mixes in results[n_layers]:
                entropy_arrays = results[n_layers][n_mixes]
                max_targets = max([len(arr) for arr in entropy_arrays]) if entropy_arrays else 0
                
                for target_idx in range(max_targets):
                    for run_idx in range(len(entropy_arrays)):
                        if target_idx < len(entropy_arrays[run_idx]):
                            detailed_data.append({
                                'Layers': n_layers,
                                'Mixes_per_Layer': n_mixes,
                                'Run': run_idx,
                                'Target_Index': target_idx,
                                'Entropy': entropy_arrays[run_idx][target_idx]
                            })
    
    df_detailed = pd.DataFrame(detailed_data)
    detailed_csv = f'{logDir}{timestamp}entropy_detailed_data.csv'
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"Detailed data saved to {detailed_csv}\n")

if __name__ == "__main__":
    main()