from random import sample, choice
from Client import Client
from Mix import Mix
from numpy.random import exponential
from Message import Message
import numpy as np
import random

global idDummies


class PoissonMix(Mix):
    def __init__(self, mix_id, simulation, position,link_based_dummies,multiple_hop_dummies, rate_mix_dummies, n_targets, corrupt, pr_mix):
        super().__init__(mix_id, simulation, position, n_targets, corrupt)
        self.pool = []  # Pool
        self.neighbors = set()
        self.link_based_dummies = link_based_dummies
        self.multiple_hop_dummies = multiple_hop_dummies
        self.rate_mix_dummies = rate_mix_dummies
        self.pr_mix = pr_mix
        self.pool_dummies = []
        if (self.link_based_dummies or self.multiple_hop_dummies) and (not self.corrupt) and self.layer != self.simulation.n_layers:
            self.env.process(self.send_dummies())

    def receive_message(self, msg):
        # stable mix
        if not self.simulation.startAttack:  # if a mix reaches a poolsize of 5 percent higher than the average,
            # the GPA can monitor the network and choose a target message
            clients = self.simulation.n_clients
            lambdaClient = self.simulation.rate_client
            average = (clients * lambdaClient * self.simulation.mu) * self.pr_mix
            var1 = len(self.pool) >= average
            print(f"var1 -> {var1}  self.pool -> {len(self.pool)}  average -> {average}")
            if self.simulation.topology == 'stratified':
                if var1 and self.layer == 1:
                    self.env.process(self.simulation.set_stable_mix(self.id - 1))
                #if all(self.simulation.stableMixL1):
                    #for i in range(len(self.simulation.stableMixL1)):
                        #self.simulation.setStableMix(i)
            elif self.simulation.topology == 'XRD':
                if var1:
                    self.env.process(self.simulation.setStableChain(self.n_chain))
                if all(self.simulation.stableChains):
                    for i in range(len(self.simulation.stableChains)):
                        self.simulation.setStableChain(i)
        # to target-fixed-msgs
        # for i in range(0, self.n_targets):
        # to target-all-msgs
        total_msgs = len(self.simulation.Log.sent_messages["MessageID"])
        while len(msg.pr_target) < total_msgs:
            msg.pr_target.append(0.0)
        while len(self.Pmix) < total_msgs:
            self.Pmix.append(0.0)
        for i in range(total_msgs):
            self.Pmix[i] += msg.pr_target[i]
        if msg.target_bool and self.simulation.printing:
            print(f'Target message arrived at mix {self.id} at time {self.env.now} and Number of messages inside '
                  f'the pool {len(self.pool)}')
        msg.next_hop_index += 1
        if msg.route[msg.next_hop_index] == None:
            # bug
            # msg.route[msg.next_hop_index] = random.choice(self.neighbors)
            # fix
            msg.route[msg.next_hop_index] = random.choice(list(self.neighbors))
        # bug
        # if msg.type == 'Real':
        # fix client-dummy
        if msg.type == 'Real'or msg.type == 'ClientDummy':
            self.pool.append(msg)
            self.env.process(self.send_msg(msg))
        elif msg.type == 'Dummy':
            if self.link_based_dummies:
                self.drop_dummies(msg)
            elif self.multiple_hop_dummies:
                if self.layer == self.simulation.n_layers:
                    self.drop_dummies(msg)
                elif self.layer != self.simulation.n_layers:
                    self.pool.append(msg)
                    self.env.process(self.send_msg(msg))
    def send_msg(self, msg):
        yield self.env.timeout(msg.delays[self.layer])
        self.update_probabilities(msg, len(self.pool))
        next_hop_index = msg.route[msg.next_hop_index]
        self.pool.remove(msg)
        self.env.process(self.simulation.attacker.relay(msg, self, next_hop_index))

    def update_probabilities(self, msg, pool_size):
        if not self.corrupt:
            # to target-fixed-msgs
            # for j in range(0, self.n_targets):
            print(f"[TARGET][update probability]Pool Size: {pool_size}")
            # to target-all-msgs
            total_msgs = len(self.simulation.Log.sent_messages["MessageID"])
            while len(msg.pr_target) < total_msgs:
                msg.pr_target.append(0.0)
            while len(self.Pmix) < total_msgs:
                self.Pmix.append(0.0)
            for j in range(total_msgs):
                msg.pr_target[j] = self.Pmix[j] / pool_size
                self.Pmix[j] = self.Pmix[j] - msg.pr_target[j]
            print(f"[TARGET][update probability][{msg.id}] {msg.target_bool}: {msg.pr_target}")

    def drop_dummies(self, msg):
        # to target-all-msgs
        self.pool.append(msg)
        # bug
        # if self.layer == self.simulation.n_layers:
        # fix client-dummy
        self.simulation.Log.dummies_dropped_end_link( msg, self.id)
        print(f"[DROP DUMMY][{msg.id}] dropped at Mix[{self.id}] Layer {self.layer}")

    def send_dummies(self):
        dummy_id = 1
        while True:
            new_message = self.create_dummies(dummy_id)
            new_message.creator = self.id
            print(f"[CREATE DUMMY][{new_message.id}] created at Mix[{self.id}] Layer {self.layer}")
            # to target-all-msgs
            total_messages = len(self.simulation.Log.sent_messages["MessageID"])
            while len(new_message.pr_target) < total_messages:
                new_message.pr_target.append(0.0)
            for i in range(total_messages):
                if i == total_messages-1:
                    print(f"[CREATE DUMMY] ID: {new_message.id} | i= {i} | total_messages-1 = {total_messages-1}")
                    new_message.pr_target[i] = float(1.0)
                else:
                    new_message.pr_target[i] = float(0.0)
            while len(self.Pmix) < total_messages:
                self.Pmix.append(0.0)
            for i in range(total_messages):
                self.Pmix[i] += new_message.pr_target[i]
            new_message.time_left = self.env.now
            self.simulation.Log.sent_messages_f(new_message)
            self.pool.append(new_message)
            yield self.env.timeout(exponential(self.rate_mix_dummies))
            dummy_id += 1
            self.env.process(self.send_msg(new_message))