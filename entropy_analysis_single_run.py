import pandas as pd
import matplotlib.pyplot as plt
import os
import configparser
from datetime import datetime

# Configuration
logDir = 'Logs/'

# Read configuration from config file
config = configparser.ConfigParser()
config.read('ConfigFile.ini')

try:
    n_layers = config.getint('TOPOLOGY', 'n_layers')
    n_mixes_per_layer = config.getint('TOPOLOGY', 'l_mixes_per_layer')
except configparser.NoOptionError as e:
    print(f"Error: {e}")
    print("Please check your ConfigFile.ini for the correct section and option names.")
    exit()

# Create plots directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Read the entropy CSV file
entropy_file = f'{logDir}{n_layers}layers_{n_mixes_per_layer}mixes_player_Entropy.csv'

if not os.path.exists(entropy_file):
    print(f"Error: {entropy_file} not found. Make sure the simulation has been run.")
else:
    # Load data
    df_entropy = pd.read_csv(entropy_file)
    send_times = df_entropy['SendTime'].values
    entropy_values = df_entropy['Entropy'].values
    
    # Create line plot
    plt.figure(figsize=(10, 6))
    plt.plot(send_times, entropy_values, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Send Time (simulation time)')
    plt.ylabel('Entropy (bits)')
    plt.title('Entropy per Target Message over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f'{logDir}entropy_{n_layers}layers_{n_mixes_per_layer}mixes_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    plt.show()