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

    def relay(self, msg, sender, receiver):
        # Choose target message
        if self.simulation.mix_type == 'pool' and len(
                self.simulation.numberrounds) > self.simulation.n_mixes_per_layer * self.simulation.n_layers:
            self.simulation.startAttack = True
        # remove stable mix condition
        # if self.var and self.simulation.startAttack and msg.next_hop_index == 1 and (
        if self.var and msg.next_hop_index == 1 and (
                msg.type == 'Real' or msg.type == 'ClientDummy'):
            # to target-fixed-msgs
            # if self.n_target_chosen_attacker < self.n_targets:
            #     for i in range(0, self.n_targets):
            #         if i == self.n_target_chosen_attacker:
            if self.n_target_chosen_attacker < len(self.simulation.Log.sent_messages["MessageID"]):
                print(f"[TARGET] ID: {msg.id} | n_target_chosen_attacker = {self.n_target_chosen_attacker} | total_messages = {len(self.simulation.Log.sent_messages['MessageID'])}")
                total_messages = len(self.simulation.Log.sent_messages["MessageID"])
                while len(msg.pr_target) < total_messages:
                    msg.pr_target.append(0.0)
                for i in range(total_messages):
                    if i == total_messages-1:
                        print(f"[TARGET] ID: {msg.id} | i= {i} | n_target_chosen_attacker = {self.n_target_chosen_attacker} | total_messages-1 = {total_messages-1}")
                        msg.pr_target[i] = float(1.0)
                    else:
                        msg.pr_target[i] = float(0.0)
                msg.target_bool = True
                self.targetMessage = msg
                print(f"[TARGET] {msg.id} n_target_chosen_attacker={self.n_target_chosen_attacker}")
                print(f"[TARGET][{msg.id}] {msg.target_bool}: {msg.pr_target}")
                # Log target information
                self.simulation.Log.log_target(
                    n_target= total_messages-1,
                    msg_id=msg.id,
                    time_left=msg.time_left,
                )
                self.n_target_chosen_attacker += 1
                if self.n_target_chosen_attacker == 1:
                    self.time_stable = self.env.now
                    if self.simulation.printing:
                        print("Network is stable at: ", self.time_stable)
                if self.simulation.printing:
                    print(f"Target message chosen: id={msg.id}, route={msg.route} at time= {self.env.now}")
                # remove stable mix condition reset
                # self.var = False
                yield self.env.timeout(2)
                self.var = True

        self.simulation.Log.log_link_load(msg, sender, receiver, self.env.now)
        yield self.env.timeout(0.05)  # 'link' delay
        receiver.receive_message(msg)
        self.checkEndSim()

    def checkEndSim(self):  # check to end simulation logic
        # to target-fixed-msgs
        # if self.env.now >= (self.simulation.SimDuration + self.simulation.burnout)and (self.simulation.n_targets == self.n_target_chosen_attacker):
        # to target-all-msgs
        if self.env.now >= (self.simulation.SimDuration + self.simulation.burnout):
            if self.simulation.printing:
                print('Simulation duration limit reached')
            self.simulation.endEvent.succeed()  # end simulation if time has expired