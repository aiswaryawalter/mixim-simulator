from Simulation import Simulation
import time
from multiprocessing import Pool
from util import Weights
import configparser

def main(rate):

    config = configparser.ConfigParser()
    config.read('ConfigFile.ini')

    topology_vars = config['TOPOLOGY']
    topology = topology_vars['type']
    fully_connected = topology_vars.getboolean('fully_connected')
    routing = topology_vars['routing']
    mix_type = config['MIXING']['mix_type']


    #Clients
    n_clients = int(config['DEFAULT']['n_clients'])
    lambda_c =  float(config['DEFAULT']['lambda_c'])
    #For Stratified Topology
    n_layer = int(config['TOPOLOGY']['n_layers'])
    n_mix_per_layer = int(config['TOPOLOGY']['l_mixes_per_layer'])
    total_n_mixes= n_layer * n_mix_per_layer
    # For cascade topology
    n_cascade = int(config['TOPOLOGY']['n_cascades'])

    mu = (int(config['TOPOLOGY']['E2E']) - (n_layer + 1)*0.05)/n_layer
    threshold = int(config['MIXING']['threshold'])
    flush_percent = float(config['MIXING']['flush_percent'])
    timeout = float(config['MIXING']['timeout'])

    # Threat Model
    corrupt_mixes = int(config['THREATMODEL']['corrupt_mixes'])
    balanced_corruption = bool(config['THREATMODEL']['balanced_corruption'])

    #Dummies
    dummies_vars = config['DUMMIES']
    client_dummies = dummies_vars.getboolean('client_dummies')
    rate_client_dummies = float(dummies_vars['rate_client_dummies'])
    link_dummies = dummies_vars.getboolean('link_based_dummies')
    multiple_hops_dummies = dummies_vars.getboolean('multiple_hop_dummies')
    rate_mix_dummies =float(dummies_vars['rate_mix_dummies'])



    weights = Weights(n_layer, n_mix_per_layer)
    simulation = Simulation(mix_type=mix_type, simDuration=50, rate_client=1/lambda_c, mu=mu, logging=True,
                            topology=topology,fully_connected= fully_connected, n_clients=n_clients, flush_percent=flush_percent, printing=True, flush_timeout=timeout, threshold=threshold, routing=routing, n_layers=n_layer,
                            n_mixes_per_layer=n_mix_per_layer,corrupt= corrupt_mixes,unifrom_corruption= balanced_corruption,probability_dist_mixes=weights,nbr_cascacdes = n_cascade, client_dummies=client_dummies,rate_client_dummies = rate_client_dummies, link_based_dummies = link_dummies, multiple_hops_dummies = multiple_hops_dummies,rate_mix_dummies = rate_mix_dummies,
                            Network_template=None)

    now = time.time()
    entropy, entropy_mean, entropy_median , entropy_q25= simulation.run()
    if simulation.printing:
        print('Simulation runtime: {}'.format(time.time() - now))

    return [entropy, entropy_mean, entropy_median , entropy_q25]

if __name__ == "__main__":
    p = Pool(processes=1, maxtasksperchild=1)
    param = [3]
    result = p.map(main,param, chunksize=1)
    table_entropy = []
    table_mean_entropy = []
    table_median_entropy = []
    table_q25_entropy = []
    table_epsilon = []
    table_delta_epsilon = []
    for item in result:
        table_entropy.append(item[0])
        table_mean_entropy.append(item[1])
        table_median_entropy.append(item[2])
        table_q25_entropy.append(item[3])
    print("Entropy", table_entropy)
    print("Mean Entropy", table_mean_entropy)
    print("Median Entropy", table_median_entropy)
    print("Quantile Entropy 0.25", table_q25_entropy)