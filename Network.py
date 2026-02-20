from PoissonMix import PoissonMix
from Pool import Pool
from TimedMix import TimedMix

import random


class Network:
    all_mixes = []
    network_dict = {}  # 1:[list of mixes in layer 1], 2:[list of mixes in layer 2], ...

    def __init__(self, mix_type, num_layers, nbr_mixes_layers, corrupt, unifrom_corruption, simulation,
                 threshold,
                 flush_percent, topology,fully_connected, flushtime, probability_dist_mixes, n_cascades,  link_based_dummies, multiple_hop_dummies, rate_mix_dummies, Network_template, numberTargets):
        self.simulation = simulation
        self.num_layers = num_layers
        self.mix_type = mix_type
        self.mixesPerLayer = nbr_mixes_layers
        self.corrupt = corrupt
        self.unifrom_corruption = unifrom_corruption
        self.env = simulation.env
        self.threshold = threshold
        self.flush_percent = flush_percent
        self.topology = topology
        self.fully_connected = fully_connected
        self.flushtime = flushtime
        self.n_cascades = n_cascades
        self.link_based_dummies = link_based_dummies
        self.multiple_hop_dummies =multiple_hop_dummies
        self.rate_mix_dummies = rate_mix_dummies
        self.probability_dist_mixes = probability_dist_mixes
        self.Network_template = Network_template
        self.numberTargets = numberTargets
        self.list_cascades = {}
        self.n_cascades = 6
        self.create_network()

    def create_network(self):
        mixnb = 1
        self.all_mixes = set()
        if self.topology == 'stratified':
            Nbr_Corruption = 0
            for layer in range(1, self.num_layers + 1):
                c = 0
                self.network_dict[layer] = []
                for _ in range(self.mixesPerLayer):
                    if self.unifrom_corruption:
                        if c < self.corrupt / self.simulation.n_layers:
                            varCorrupt = True
                            c += 1
                        else:
                            varCorrupt = False
                    else:
                        if Nbr_Corruption < self.corrupt:
                            varCorrupt = random.choice([True, False])
                            if varCorrupt:
                                Nbr_Corruption += 1
                        else:
                            varCorrupt = False
                    mix = self.get_mixnode(self.mix_type, mixnb, layer, self.numberTargets, varCorrupt,
                                           self.probability_dist_mixes[layer - 1][_])
                    self.all_mixes.add(mix)
                    self.network_dict[layer] += [mix]
                    mixnb += 1

            for mix in self.all_mixes:
                if self.fully_connected:
                    if mix.layer + 1 in self.network_dict:  # last mix doesn't need neighbors
                        mix.neighbors = self.network_dict[mix.layer + 1]
                    if mix.layer == self.simulation.n_layers:
                        mix.neighbors = self.network_dict[1]
                else:
                    pass
                    #for mix in self.MixesAll:
                        #if mix.id == 1:
                            #mix.neighbors = []
                            #mix.neighbors.append(self.LayerDict[mix.layer + 1][0])
                            #mix.neighbors.append(self.LayerDict[mix.layer + 1][1])
        elif self.topology == 'XRD':
            mixnb = 1
            for n in range(1, 1 + self.n_cascades):
                cascade = []
                for m in range(self.num_layers):
                    varCorrupt = False
                    mix = self.get_mixnode(self.mix_type, mixnb, m + 1, self.numberTargets, varCorrupt,
                                           1 / self.n_cascades)
                    mix.n_chain = n
                    self.all_mixes.add(mix)
                    mixnb += 1
                    cascade.append(mix)
                self.list_cascades[n] = cascade
            for n, list in self.list_cascades.items():
                print('Chain number', n, ':', list)

    def get_mixnode(self, mix_type, id, position, numberTargets, corrupt, weight_mix):
        if mix_type == 'poisson':
            return PoissonMix(id, self.simulation, position, self.link_based_dummies,self.multiple_hop_dummies, self.rate_mix_dummies,
                              numberTargets, corrupt, weight_mix)
        elif mix_type == 'pool':
            return Pool(id, self.simulation, position, self.threshold, self.flush_percent, numberTargets,
                        corrupt, weight_mix)
        elif mix_type == 'time':
            return TimedMix(id, self.simulation, position, self.flushtime, numberTargets, corrupt,
                            weight_mix)

    def odd(self, number):
        return number % 2 == 1
