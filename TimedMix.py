
from random import sample
from Client import Client
from Mix import Mix


class TimedMix(Mix):

    def __init__(self, mix_id, simulation, position, flush_timeout,n_targets, corrupt,weight_mix):
        super().__init__(mix_id, simulation,position,n_targets, corrupt)
        self.pool = []
        self.flush_timeout = flush_timeout
        self.weight_mix = weight_mix
        self.neighbors = set()
        self.env.process(self.flush())

    def receive_message(self, msg):
        if not self.simulation.startAttack:
            var1 = self.simulation.env.now > self.flush_timeout  # first time this mix needs to flush messages
            if var1 and self.layer ==1:
                self.env.process(self.simulation.set_stable_mix(self.id - 1))
            if all(self.simulation.stableMixL1):
                for i in range(self.simulation.n_mixes_per_layer):
                    self.simulation.set_stable_mix(i)
        for i in range(0, self.n_targets):
            self.Pmix[i] += msg.pr_target[i]
        if msg.target_bool and self.simulation.printing:
            print(
                f'Target message arrived at mix {self.id} at time {self.env.now} and size of the pool{len(self.pool)}')
        msg.next_hop_index += 1
        self.pool.append(msg)

    def flush(self):
        while True:
            yield self.env.timeout(self.flush_timeout)
            for message in self.pool:
                self.update_probabilities(message)
                next_hop_index = message.route[message.next_hop_index]
                self.env.process(self.simulation.attacker.relay(message, next_hop_index))

            self.pool.clear()

    def update_probabilities(self, message):
        if not self.corrupt:
            for i in range(0, self.n_targets):
                message.pr_target[i] = self.Pmix[i] / len(self.pool)
                self.Pmix[i] = self.Pmix[i] - message.pr_target[i]
