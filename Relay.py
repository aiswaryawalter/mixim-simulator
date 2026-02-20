link_delay = [0.01, 0.1]


import numpy as np


class Attacker:

    def __init__(self, simulation, n_targets):
        self.simulation = simulation
        self.var = True  # if target message should still be chosen
        self.env = self.simulation.env
        self.targetMessage = None
        self.n_target_chosen_attacker = 0
        self.n_targets = n_targets
        self.time_stable = 0.0

    def relay(self, msg, receiver):
        # Choose target message
        if self.simulation.mix_type == 'pool' and len(
                self.simulation.numberrounds) > self.simulation.n_mixes_per_layer * self.simulation.n_layers:
            self.simulation.startAttack = True
        if self.var and self.simulation.startAttack and msg.next_hop_index == 1 and (
                msg.type == 'Real' or msg.type == 'ClientDummy'):
            if self.n_target_chosen_attacker < self.n_targets:
                for i in range(0, self.n_targets):
                    if i == self.n_target_chosen_attacker:
                        msg.pr_target[i] = float(1.0)
                    else:
                        msg.pr_target[i] = float(0.0)
                msg.target_bool = True
                self.targetMessage = msg
                self.n_target_chosen_attacker += 1
                if self.n_target_chosen_attacker == 1:
                    self.time_stable = self.env.now
                    if self.simulation.printing:
                        print("Network is stable at: ", self.time_stable)
                if self.simulation.printing:
                    print(f"Target message chosen: id={msg.id}, route={msg.route} at time= {self.env.now}")
                self.var = False
                yield self.env.timeout(2)
                self.var = True

        yield self.env.timeout(0.05)  # 'link' delay
        receiver.receive_message(msg)
        self.checkEndSim()

    def checkEndSim(self):  # check to end simulation logic
        if self.env.now >= (self.simulation.SimDuration + self.simulation.burnout)and (self.simulation.n_targets == self.n_target_chosen_attacker):
            if self.simulation.printing:
                print('Simulation duration limit reached')
            self.simulation.endEvent.succeed()  # end simulation if time has expired