import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def load_results(timestamp):
    """Load saved results using timestamp"""
    logDir = 'Logs/'
    
    # Load raw results
    results_pickle = f'{logDir}{timestamp}results_raw.pkl'
    with open(results_pickle, 'rb') as f:
        results = pickle.load(f)
    
    # Load detailed data CSV
    detailed_csv = f'{logDir}{timestamp}entropy_detailed_data.csv'
    df_detailed = pd.read_csv(detailed_csv)
    
    return results, df_detailed

def plot_entropy_evolution(results, timestamp, layers_list, mixes_list, logDir):
    """Generate all entropy evolution plots"""
    
    # Plot 1: Entropy evolution for layers = 3
    plt.figure(figsize=(14, 7))
    if 3 in results:
        for n_mixes in mixes_list:
            if n_mixes in results[3]:
                entropy_arrays = results[3][n_mixes]
                max_targets = max([len(arr) for arr in entropy_arrays])
                avg_entropy_per_target = []
                
                for target_idx in range(max_targets):
                    values = [entropy_arrays[run][target_idx] for run in range(len(entropy_arrays)) 
                             if target_idx < len(entropy_arrays[run])]
                    if values:
                        avg_entropy_per_target.append(np.mean(values))
                
                plt.plot(range(len(avg_entropy_per_target)), avg_entropy_per_target, 
                        marker='o', label=f'{n_mixes} Mixes', linewidth=2, markersize=6, alpha=0.8)
    
    plt.xlabel('Target Message Index (Time Progression)', fontsize=13)
    plt.ylabel('Average Entropy (bits)', fontsize=13)
    plt.title('Entropy Evolution During Simulation (3 Layers)', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1_filename = f'{logDir}{timestamp}_entropy_evolution_3layers.png'
    plt.savefig(plot1_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot1_filename}")
    plt.close()

    # Plot 2: Entropy evolution for layers = 4
    plt.figure(figsize=(14, 7))
    if 4 in results:
        for n_mixes in mixes_list:
            if n_mixes in results[4]:
                entropy_arrays = results[4][n_mixes]
                max_targets = max([len(arr) for arr in entropy_arrays])
                avg_entropy_per_target = []
                
                for target_idx in range(max_targets):
                    values = [entropy_arrays[run][target_idx] for run in range(len(entropy_arrays)) 
                             if target_idx < len(entropy_arrays[run])]
                    if values:
                        avg_entropy_per_target.append(np.mean(values))
                
                plt.plot(range(len(avg_entropy_per_target)), avg_entropy_per_target, 
                        marker='s', label=f'{n_mixes} Mixes', linewidth=2, markersize=6, alpha=0.8)
    
    plt.xlabel('Target Message Index (Time Progression)', fontsize=13)
    plt.ylabel('Average Entropy (bits)', fontsize=13)
    plt.title('Entropy Evolution During Simulation (4 Layers)', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2_filename = f'{logDir}{timestamp}_entropy_evolution_4layers.png'
    plt.savefig(plot2_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot2_filename}")
    plt.close()

    # Plot 3: Entropy evolution for layers = 5
    plt.figure(figsize=(14, 7))
    if 5 in results:
        for n_mixes in mixes_list:
            if n_mixes in results[5]:
                entropy_arrays = results[5][n_mixes]
                max_targets = max([len(arr) for arr in entropy_arrays])
                avg_entropy_per_target = []
                
                for target_idx in range(max_targets):
                    values = [entropy_arrays[run][target_idx] for run in range(len(entropy_arrays)) 
                             if target_idx < len(entropy_arrays[run])]
                    if values:
                        avg_entropy_per_target.append(np.mean(values))
                
                plt.plot(range(len(avg_entropy_per_target)), avg_entropy_per_target, 
                        marker='^', label=f'{n_mixes} Mixes', linewidth=2, markersize=6, alpha=0.8)
    
    plt.xlabel('Target Message Index (Time Progression)', fontsize=13)
    plt.ylabel('Average Entropy (bits)', fontsize=13)
    plt.title('Entropy Evolution During Simulation (5 Layers)', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot3_filename = f'{logDir}{timestamp}_entropy_evolution_5layers.png'
    plt.savefig(plot3_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot3_filename}")
    plt.close()

def main():
    # Specify the timestamp of the results to load
    timestamp = input("Enter the timestamp of results to plot (e.g., 20260225_143052): ")
    
    logDir = 'Logs/'
    results_pickle = f'{logDir}{timestamp}results_raw.pkl'
    
    if not os.path.exists(results_pickle):
        print(f"Error: Results file {results_pickle} not found!")
        return
    
    print(f"Loading results from timestamp {timestamp}...")
    results, df_detailed = load_results(timestamp)
    
    # Get parameters from the data
    layers_list = sorted(df_detailed['Layers'].unique())
    mixes_list = sorted(df_detailed['Mixes_per_Layer'].unique())
    
    print(f"Found data for:")
    print(f"  Layers: {layers_list}")
    print(f"  Mixes per layer: {mixes_list}")
    
    print(f"\nGenerating plots...")
    plot_entropy_evolution(results, timestamp, layers_list, mixes_list, logDir)
    
    print(f"\nAll plots saved to {logDir} with prefix {timestamp}_")

if __name__ == "__main__":
    main()