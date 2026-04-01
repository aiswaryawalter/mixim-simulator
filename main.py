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
    # \\mu = 0.13 for stratified and mu = 0.9 for free route

    # mu = float(config['MIXING']['mu'])
    print(f"mu = {mu}")
    threshold = int(config['MIXING']['threshold'])
    flush_percent = float(config['MIXING']['flush_percent'])
    timeout = float(config['MIXING']['timeout'])
    # to stop sending real messages towards the end a certain percentage of the total simulation time
    stop_real_msgs_percent = float(config['MIXING']['stop_real_msgs_percent'])
    # to wait for certain percent of sent messages to be delivered before stopping the simulation
    msg_delivery_percent = float(config['MIXING']['msg_delivery_percent'])

    # Threat Model
    corrupt_mixes = int(config['THREATMODEL']['corrupt_mixes'])
    balanced_corruption = config['THREATMODEL'].getboolean('balanced_corruption')
    # balanced_corruption = bool(config['THREATMODEL']['balanced_corruption'])

    #Dummies
    dummies_vars = config['DUMMIES']
    client_dummies = dummies_vars.getboolean('client_dummies')
    # remove: rate_client_dummies. computing the rate_client_dummies based on rho
    # rate_client_dummies = float(dummies_vars['rate_client_dummies'])
    link_dummies = dummies_vars.getboolean('link_based_dummies')
    multiple_hops_dummies = dummies_vars.getboolean('multiple_hop_dummies')
    # remove: rate_mix_dummies. computing the rate_mix_dummies based on rho
    # rate_mix_dummies =float(dummies_vars['rate_mix_dummies'])

    # add: compute rate_client_dummies and rate_mix_dummies based on rho
    rho = float(dummies_vars['rho'])
    rate_client_dummies = 0.0
    rate_mix_dummies = 0.0

    M = n_mix_per_layer
    L = n_layer
    C = n_clients

    if client_dummies:
        # lambda_cd = (rho * M^2) / C
        lambda_cd = (rho * (M ** 2)) / C
        if lambda_cd <= 0:
            raise ValueError("lambda_cd must be > 0")
        rate_client_dummies = 1.0 / lambda_cd
        print(f"[DUMMIES][CLIENT] rho={rho}, lambda_cd={lambda_cd}, rate_client_dummies={rate_client_dummies}")

    elif link_dummies:
        # lambda_ld = rho * M
        lambda_ld = rho * M
        if lambda_ld <= 0:
            raise ValueError("lambda_ld must be > 0")
        rate_mix_dummies = 1.0 / lambda_ld
        print(f"[DUMMIES][LINK] rho={rho}, lambda_ld={lambda_ld}, rate_mix_dummies={rate_mix_dummies}")

    elif multiple_hops_dummies:
        # lambda_md = ((L-1) * M * rho) / sum_{r=1}^{L-1}(L-r) = 2*M*rho/L
        denom = sum((L - r) for r in range(1, L))
        lambda_md = ((L - 1) * M * rho) / denom
        if lambda_md <= 0:
            raise ValueError("lambda_md must be > 0")
        rate_mix_dummies = 1.0 / lambda_md
        print(f"[DUMMIES][MULTIHOP] rho={rho}, lambda_md={lambda_md}, rate_mix_dummies={rate_mix_dummies}")



    weights = Weights(n_layer, n_mix_per_layer)
    simulation = Simulation(mix_type=mix_type, simDuration=20, rate_client=1/lambda_c, mu=mu, logging=True,
                            topology=topology,fully_connected= fully_connected, n_clients=n_clients, flush_percent=flush_percent, printing=True, flush_timeout=timeout, threshold=threshold, routing=routing, n_layers=n_layer,
                            n_mixes_per_layer=n_mix_per_layer,corrupt= corrupt_mixes,unifrom_corruption= balanced_corruption,probability_dist_mixes=weights,nbr_cascacdes = n_cascade, client_dummies=client_dummies,rate_client_dummies = rate_client_dummies, link_based_dummies = link_dummies, multiple_hops_dummies = multiple_hops_dummies,rate_mix_dummies = rate_mix_dummies,
                            Network_template=None,
                            stop_real_msgs_percent = stop_real_msgs_percent, 
                            msg_delivery_percent = msg_delivery_percent)

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