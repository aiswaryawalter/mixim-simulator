
class Log:
    def __init__(self):
        self.sent_messages = {"MessageID": [], "MessageType": [], "MessageTimeLeft" :[], "MessageDelay": [], "MessageRoute" :[]}
        self.received_messages = {"MessageID": [], "MessageType": [], "MessageTimeLeft" :[],"MessageTimeReceived":[], "MessageDelay": [], "MessageRoute" :[],"MessageTarget" : [] }
        self.dummy_messages = {"DroppingNode":[],"DummyID": [], "DummyType": [], "DummyTimeLeft" :[], "DummyDelay": [], "DummyRoute" :[], "DummyPr":[]}
        self.target_messages = {"TargetIndex": [], "MessageID": [], "TimeLeft": []}
        self.link_load = {
            'Time': [],
            'MessageID': [],
            'MessageType': [],  # 'Real', 'Dummy', 'ClientDummy'
            'SenderID': [],
            'ReceiverID': [],
            'SenderLayer': [],
            'ReceiverLayer': [],
            'IsMixMix': []
        }

    def dummies_dropped_end_link(self, dummy, dropping_node):
        self.dummy_messages["DroppingNode"].append(dropping_node)
        self.dummy_messages["DummyID"].append(dummy.id)
        self.dummy_messages["DummyType"].append(dummy.type)
        self.dummy_messages["DummyTimeLeft"].append(dummy.time_left)
        self.dummy_messages["DummyDelay"].append(dummy.delays)
        self.dummy_messages["DummyRoute"].append(dummy.route)
        self.dummy_messages["DummyPr"].append(dummy.pr_target)

    def sent_messages_f(self, msg):
        self.sent_messages["MessageID"].append(msg.id)
        self.sent_messages["MessageType"].append(msg.type)
        self.sent_messages["MessageTimeLeft"].append(msg.time_left)
        self.sent_messages["MessageDelay"].append(msg.delays)
        self.sent_messages["MessageRoute"].append(msg.route)

    def received_messages_f(self, msg):
        self.received_messages["MessageID"].append(msg.id)
        self.received_messages["MessageType"].append(msg.type)
        self.received_messages["MessageTimeLeft"].append(msg.time_left)
        self.received_messages["MessageTimeReceived"].append(msg.timeReceived)
        self.received_messages["MessageDelay"].append(msg.delays)
        self.received_messages["MessageRoute"].append(msg.route)
        self.received_messages["MessageTarget"].append(msg.pr_target)
    
    def log_target(self, n_target, msg_id, time_left):
        self.target_messages["TargetIndex"].append(n_target)
        self.target_messages["MessageID"].append(msg_id)
        self.target_messages["TimeLeft"].append(time_left)
    
    def log_link_load(self, message, sender, receiver, now):
        """Log a message traversal across a single hop."""
        sender_layer = getattr(sender, 'layer', None)  # None for clients
        receiver_layer = getattr(receiver, 'layer', None)
        is_mix_mix = (sender_layer is not None) and (receiver_layer is not None)
        
        self.link_load['Time'].append(now)
        self.link_load['MessageID'].append(message.id)
        self.link_load['MessageType'].append(message.type)
        self.link_load['SenderID'].append(sender.id)
        self.link_load['ReceiverID'].append(receiver.id)
        self.link_load['SenderLayer'].append(sender_layer)
        self.link_load['ReceiverLayer'].append(receiver_layer)
        self.link_load['IsMixMix'].append(is_mix_mix)
