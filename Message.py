class Message:
    def __init__(self, id, type, sender, route, delays, pr_target, target_bool):
        self.id = "%d_%d" % (sender.id, id)
        self.type = type  # Dummy or Real packet
        self.sender = sender  # sender object
        self.route = route  # e.g. [S1, mix1, mix2, mix3, R5]
        self.delays = delays  # list of delays at 3 nodes
        self.pr_target = pr_target  # probability of this message being the target message
        self.target_bool = target_bool  # True if this message is a target message
        self.time_left = 0
        self.next_hop_index = 1
