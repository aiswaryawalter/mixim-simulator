from random import sample
from Client import Client
from Mix import Mix

flushed = False  # True if a mix has flushed since the beginning of the simulation


class Pool(Mix):

    def __init__(self, mix_id, simulation, position,threshold, flush_percent,n_targets, corrupt, pr_mix):
        super().__init__(mix_id, simulation, position, n_targets, corrupt)

        self.pool = []
        self.pr_mix= pr_mix
        self.neighbors = []  # mixes in next layer used if routing == 'hopbyhop'
        self.threshold = threshold  # pool threshold
        self.flush_percent = flush_percent
        self.round = 0

    def receive_message(self, msg):
        msg.next_hop_index += 1
        self.pool.append(msg)
        for i in range(0, self.n_targets):
            self.Pmix[i] += msg.pr_target[i]
        if msg.target_bool and self.simulation.printing:
            print(
                f'Target message arrived at mix {self.id} at time {self.env.now} and size of the pool{len(self.pool)}')
        if len(self.pool) >= self.threshold:
            self.flush()
            self.round +=1
            self.simulation.numberrounds.append(self.round)
            for i in range(self.simulation.n_mixes_per_layer):
                self.simulation.set_stable_mix(i)

    def flush(self):
        flush_amount = int(self.flush_percent * self.threshold)
        flushing_list = sample(self.pool, k=flush_amount)

        for message in flushing_list:
            self.update_probabilities(message)
            if not isinstance(message.route[message.next_hop_index], Client) and message.route[message.next_hop_index] is None:
                # not the last mix, for hop by hop routing
                message.route[message.next_hop_index] = sample(self.neighbors, k=1)[0]

            next_hop_index = message.route[message.next_hop_index]
            self.pool.remove(message)
            self.env.process(self.simulation.attacker.relay(message, next_hop_index))

    def update_probabilities(self, msg):
        if not self.corrupt:
            for j in range(0, self.n_targets):
                msg.pr_target[j] = self.Pmix[j] / len(self.pool)
                self.Pmix[j] = self.Pmix[j] - msg.pr_target[j]