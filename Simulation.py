import simpy
import Client
from Network import Network
import pandas as pd
import numpy as np
from Relay import Attacker
from Log import Log
from util import XRD_New
import os

DEFAULT_TOPOLOGY = 'stratified'
logDir = 'Logs/'
if not os.path.exists(logDir):
    os.makedirs(logDir)


class Simulation(object):

    def __init__(self, mix_type, simDuration, rate_client, mu, logging, topology, fully_connected, n_clients,
                 flush_percent, printing, flush_timeout, threshold, routing, n_layers,
                 n_mixes_per_layer, corrupt, unifrom_corruption, probability_dist_mixes, nbr_cascacdes, client_dummies,
                 rate_client_dummies, link_based_dummies, multiple_hops_dummies, rate_mix_dummies, Network_template, 
                msg_delivery_percent):

        self.Log = Log()
        self.logs = []
        self.logging = logging
        self.printing = printing
        self.topology = topology
        self.n_cascades = nbr_cascacdes
        self.fully_connected = fully_connected
        self.flush_percent = flush_percent
        # to wait for certain percent of sent messages to be delivered before stopping the simulation
        self.msg_delivery_percent = msg_delivery_percent

        self.client_dummies = client_dummies
        self.rate_client_dummies = rate_client_dummies
        self.link_based_dummies = link_based_dummies
        self.multiple_hop_dummies = multiple_hops_dummies
        self.rate_mix_dummies = rate_mix_dummies

        self.n_clients = n_clients
        self.clientsSet = set()
        self.rate_client = rate_client  # average delay between messages being sent from client
        self.threshold = threshold
        self.mu = mu  # average delay at poisson mixes
        self.n_layers = n_layers
        self.n_mixes_per_layer = n_mixes_per_layer
        self.corrupt = corrupt
        self.probability_dist_mixes = probability_dist_mixes
        self.unifrom_corruption = unifrom_corruption
        self.mix_type = mix_type
        self.routing = routing
        self.env = simpy.Environment()
        self.SimDuration = simDuration
        # original: burnout = 10. but it not used anywhere other than in the end simulation logic
        # self.burnout = 10
        self.burnout = 0
        self.flush_timeout = flush_timeout
        self.n_targets = 0
        self.MsgsDropped = []

        self.dummyID = 0
        time_stable = ((1 / self.rate_client) / self.n_layers) * self.mu + 2
        if self.mix_type == 'poisson':
            self.n_targets = int(((self.SimDuration - time_stable) ) / 2)
            print(f"n_targets={self.n_targets}")
        else:
            self.n_targets = int((self.SimDuration - self.flush_timeout - 1) / 4)
        self.network = Network(self.mix_type, self.n_layers, self.n_mixes_per_layer, self.corrupt,
                               self.unifrom_corruption, self, self.threshold,
                               self.flush_percent, self.topology, fully_connected, self.flush_timeout,
                               self.probability_dist_mixes,
                               self.n_cascades, self.link_based_dummies, self.multiple_hop_dummies,
                               self.rate_mix_dummies,
                               Network_template, self.n_targets)

        self.set_clients(self.probability_dist_mixes, self.n_targets, self.client_dummies, self.rate_client_dummies,
                         self.Log)
        # self.stableMix = [False for i in range(self.n_mixes_per_layer*self.n_layers)]  # only start attack after mixes are stable
        self.stableChains = [False for i in range(1, 1 + 6)]  # only start attack after chains are stable
        self.stableMixL1 = [False for i in range(self.n_mixes_per_layer)]  # only start attack after mixes are stable
        self.attacker = Attacker(self, self.n_targets)  # attacker/relay object
        self.endEvent = self.env.event()  # event that triggers the end of the simulation
        self.TargetMessageEnd = False  # if target message has reached the end client
        self.startAttack = False  # if the attacker is allowed to choose a target message
        self.NumberMsgsDropped = 0
        self.numberrounds = []

    def set_stable_mix(self, index):
        print(f"[{self.env.now}] Entered set_stable_mix for index={index}")
        if self.mix_type == 'pool':
            yield self.env.timeout(10)
            self.startAttack = True
        elif self.mix_type == 'time':
            yield self.env.timeout(self.flush_timeout + 5)
            self.startAttack = True
        self.stableMixL1[index] = True
        if all(self.stableMixL1):
            yield self.env.timeout(2)
            self.startAttack = True
        print(f"[{self.env.now}] set_stable_mix done => startAttack={self.startAttack}")


    def set_stable_chain(self, position):
        if self.mix_type == 'pool':
            yield self.env.timeout(10)
            self.startAttack = True
        self.stableChains[position - 1] = True
        if all(self.stableChains):
            yield self.env.timeout(2)
            self.startAttack = True

    def set_clients(self, probabilityDistribution, n_targets, client_dummies, rate_client_dummies, Log):
        if self.topology == 'stratified':
            for client_no in range(self.n_clients):
                client = Client.Client(self, client_no, self.network.network_dict, self.rate_client, self.mu,
                                       probabilityDistribution, n_targets, client_dummies, rate_client_dummies, Log)
                self.clientsSet.add(client)
            for client in self.clientsSet:
                client.other_clients = self.clientsSet - {client}
        elif self.topology == 'XRD':
            groups_lists = XRD_New(self.network.list_cascades)
            n_group_client = self.n_clients // len(groups_lists)
            for client_id in range(n_group_client):
                client = Client.Client(self, client_id, groups_lists[0], self.rate_client, self.mu,
                                       probabilityDistribution, n_targets, client_dummies, Log)
                self.clientsSet.add(client)
            for n_client in range(n_group_client, n_group_client * 2):
                client = Client.Client(self, n_client, groups_lists[1], self.rate_client, self.mu,
                                       probabilityDistribution, n_targets, client_dummies, Log)
                self.clientsSet.add(client)
            for n_client in range(n_group_client * 2, n_group_client * 3):
                client = Client.Client(self, n_client, groups_lists[2], self.rate_client, self.mu,
                                       probabilityDistribution, n_targets, client_dummies, Log)
                self.clientsSet.add(client)
            for n_client in range(n_group_client * 3, self.n_clients):
                client = Client.Client(self, n_client, groups_lists[3], self.rate_client, self.mu,
                                       probabilityDistribution, n_targets, client_dummies, Log)
                self.clientsSet.add(client)

            for client in self.clientsSet:
                client.otherClients = self.clientsSet - {client}

    def aggregate_link_load(self, df_link_load):
        """
        Build per-transition summary (L1->L2, L2->L3, ...)
        and append one global mix-mix summary row (ALL->ALL).
        """
        out_cols = [
            'FromLayer', 'ToLayer', 'TotalMsgs', 'RealMsgs', 'DummyMsgs',
            'NumLinks', 'AvgMsgsPerLink', 'AvgRealPerLink', 'AvgDummyPerLink',
            'RatePerLinkPerTime', 'RateRealPerLinkPerTime', 'RateDummyPerLinkPerTime'
        ]

        # No traffic logged
        if df_link_load.empty:
            return pd.DataFrame(columns=out_cols)

        # Keep only mix->mix hops
        df_mix = df_link_load[df_link_load['IsMixMix'] == True].copy()
        if df_mix.empty:
            return pd.DataFrame(columns=out_cols)

        # Classify message type
        df_mix['IsReal'] = (df_mix['MessageType'] == 'Real')
        df_mix['IsDummy'] = (df_mix['MessageType'] != 'Real')  # Dummy + ClientDummy

        # Layerwise transition summary
        summary = df_mix.groupby(['SenderLayer', 'ReceiverLayer'], as_index=False).agg(
            TotalMsgs=('MessageID', 'count'),
            RealMsgs=('IsReal', 'sum'),
            DummyMsgs=('IsDummy', 'sum')
        )

        summary = summary.rename(columns={
            'SenderLayer': 'FromLayer',
            'ReceiverLayer': 'ToLayer'
        })

        links_per_transition = self.n_mixes_per_layer ** 2
        summary['NumLinks'] = links_per_transition
        summary['AvgMsgsPerLink'] = summary['TotalMsgs'] / summary['NumLinks']
        summary['AvgRealPerLink'] = summary['RealMsgs'] / summary['NumLinks']
        summary['AvgDummyPerLink'] = summary['DummyMsgs'] / summary['NumLinks']
        summary['RatePerLinkPerTime'] = summary['AvgMsgsPerLink'] / self.SimDuration
        summary['RateRealPerLinkPerTime'] = summary['AvgRealPerLink'] / self.SimDuration
        summary['RateDummyPerLinkPerTime'] = summary['AvgDummyPerLink'] / self.SimDuration

        # Global mix-mix summary across all transitions
        total_msgs_all = int(df_mix.shape[0])
        real_msgs_all = int(df_mix['IsReal'].sum())
        dummy_msgs_all = int(df_mix['IsDummy'].sum())

        # For stratified fully connected: (L-1) transitions, each has M^2 directed links
        total_mixmix_links = (self.n_layers - 1) * (self.n_mixes_per_layer ** 2)

        overall_row = pd.DataFrame([{
            'FromLayer': 'ALL',
            'ToLayer': 'ALL',
            'TotalMsgs': total_msgs_all,
            'RealMsgs': real_msgs_all,
            'DummyMsgs': dummy_msgs_all,
            'NumLinks': total_mixmix_links,
            'AvgMsgsPerLink': total_msgs_all / total_mixmix_links,
            'AvgRealPerLink': real_msgs_all / total_mixmix_links,
            'AvgDummyPerLink': dummy_msgs_all / total_mixmix_links,
            'RatePerLinkPerTime': (total_msgs_all / total_mixmix_links) / self.SimDuration,
            'RateRealPerLinkPerTime': (real_msgs_all / total_mixmix_links) / self.SimDuration,
            'RateDummyPerLinkPerTime': (dummy_msgs_all / total_mixmix_links) / self.SimDuration

        }])

        return pd.concat([summary, overall_row], ignore_index=True)

    def run(self, time=None):
        # Print statements and results from here
        if self.printing:
            print('\n')
            print('----------Simulation Data----------')
            print('Topology: {}'.format(self.topology))
            print('Routing strategy: {}'.format(self.routing))
            print('Mix type: {}'.format(self.mix_type))
            print('Layers: {}, amount of mixes per layer: {}'.format(self.n_layers, self.n_mixes_per_layer))
            print(
                'Amount of clients: {}, average delay between 2 messages: {}'.format(self.n_clients, self.rate_client))
            l = ''
            index = 0
            if self.topology == 'stratified':
                for layer in self.network.network_dict:
                    k = 'Layer {}: [ '.format(layer)
                    for mixnb in range(self.n_mixes_per_layer):
                        k += str(self.network.network_dict[layer][mixnb])
                        if self.network.network_dict[layer][mixnb].corrupt:
                            index += 1
                    l += k + ']'
                    l += '\n'
                print(l)
            elif self.topology == 'cascade':
                k2 = 'Cascade'
                for cascade in self.network.list_cascades:
                    for mixnb in range(3):
                        k2 += str(cascade[mixnb].id)
                    l += k2 + ']'
                    l += '\n'
            print('----------Starting Simulation----------')
        if self.printing:
            print('Topology: {}'.format(self.topology))
        if time is None:
            try:
                self.env.run(until=self.endEvent)
            except RuntimeError as e:
                no_events_left = 'No scheduled events left' in str(e)
                if no_events_left and not self.endEvent.triggered:
                    if self.printing:
                        sent_types = self.Log.sent_messages["MessageType"]
                        recv_types = self.Log.received_messages["MessageType"]

                        real_sent = sum(1 for t in sent_types if t == 'Real')
                        real_received = sum(1 for t in recv_types if t == 'Real')
                        pct = 100.0 * real_received / real_sent
                        print(f'{real_received}/{real_sent} real msgs delivered at ({pct:.2f}%)')
            
                        print(f'[{self.env.now}]No scheduled events left before endEvent; finishing with current logs.')
                else:
                    raise
        else:
            self.env.run(until=time)

        if self.printing:
            print('----------Simulation Ended---------')
            print('\n')

        # Data from Clients(senders and receivers)
        df_sent_messages = pd.DataFrame(self.Log.sent_messages)
        df_received_messages = pd.DataFrame(self.Log.received_messages)
        df_dummies_messages = pd.DataFrame(self.Log.dummy_messages)
        df_targets = pd.DataFrame(self.Log.target_messages)
        df_link_load = pd.DataFrame(self.Log.link_load)
        df_link_summary = self.aggregate_link_load(df_link_load)

        if self.logging:
            df_sent_messages.to_csv(f'{logDir}SentMessages.csv')
            df_received_messages.to_csv(f'{logDir}ReceivedMessages.csv')
            df_dummies_messages.to_csv(f'{logDir}DummyMessages.csv')
            df_link_load.to_csv(f'{logDir}LinkLoad.csv')
            df_link_summary.to_csv(f'{logDir}LinkSummary.csv')
            df_targets.to_csv(f'{logDir}{self.n_layers}layers_{self.n_mixes_per_layer}mixes_player_Targets.csv', index=False)
        else:
            pass

        entropy = []
        total_msgs = len(self.Log.received_messages["MessageID"])
        # to target-fixed-msgs
        # for i in range(0, self.n_targets):
        for i in range(0, total_msgs):
            entropy.append(0.0)
        # column_sums = [0.0 for _ in range(self.n_targets)]

        tableProb = df_received_messages['MessageTarget'].to_numpy(copy=True)
        # to target-fixed-msgs
        # for j in range(0, self.n_targets):
        #     for m in range(len(tableProb)):
        #         p = float(tableProb[m][j])
        #         column_sums[j] += p
        #         if tableProb[m][j] != 0:
        #             entropy[j] += - tableProb[m][j] * np.log2(tableProb[m][j])
        # print("column_sums:", column_sums)
        for m in range(len(tableProb)):
            if len(tableProb[m]) < total_msgs:
                # Extend the row with zeros
                tableProb[m] = list(tableProb[m]) + [0.0] * (total_msgs - len(tableProb[m]))
        
        # for j in range(0, total_msgs):
        #     for m in range(len(tableProb)):
        #         if tableProb[m][j] != 0:
        #             entropy[j] += - tableProb[m][j] * np.log2(tableProb[m][j])

        # to target-all-msgs
        # to compute entropy row-wise (for each message) instead of column-wise (for each target)
        for m in range(len(tableProb)):
            row = np.array(tableProb[m], dtype=float)
            row_sum = row.sum()
            # print(f"Row sum for row {m}: {row_sum}")

            if np.isclose(row_sum, 1.0, atol=0.3):
                row_entropy = 0.0
                for p in row:
                    if p != 0:
                        row_entropy += -p * np.log2(p)
                # print(f"Entropy of {m}: {row_entropy}")
                entropy[m] = row_entropy
            else:
                # print(f"Entropy of {m}: 0.0 (row sum is not 1.0)")
                entropy[m] = 0.0 

  
        dict_entropy = {'Entropy': entropy}
        df_entropy = pd.DataFrame(dict_entropy)
        df_entropy['MessageID'] = df_received_messages['MessageID'].to_numpy(copy=True)
        df_entropy['TimeLeft'] = df_received_messages['MessageTimeLeft'].to_numpy(copy=True)
        df_entropy['TimeReceived'] = df_received_messages['MessageTimeReceived'].to_numpy(copy=True)
        df_entropy.to_csv(f'{logDir}{self.n_layers}layers_{self.n_mixes_per_layer}mixes_player_Entropy.csv')

        entropy_mean = np.mean(entropy)
        try:
            entropy_median = np.median(entropy)
            entropy_q25 = np.quantile(entropy, .25)
        except:
            entropy_median = 0
            entropy_q25 = 0

        sum_delays = 0
        for re, le in zip(self.Log.received_messages["MessageTimeReceived"],
                          self.Log.received_messages["MessageTimeLeft"]):
            sum_delays += (re - le)
        average_delay = sum_delays / len(self.Log.received_messages["MessageTimeReceived"])
        if self.printing:
            print('----------Simulation Stats----------')
            # print('Average Latency: {}'.format(latency))
            print("Number of targets chosen", self.n_targets)
            print('Number of Real messages generated', len(self.Log.sent_messages["MessageID"]))
            print('Number of Real messages Received', len(self.Log.received_messages["MessageID"]))
            print('Number of Dummy messages dropped', len(self.Log.dummy_messages["DummyID"]))
            print("Average delay per message", average_delay)
            print('-------------------------------------')

        return entropy, entropy_mean, entropy_median, entropy_q25
