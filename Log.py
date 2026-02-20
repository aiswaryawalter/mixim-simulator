
class Log:
    def __init__(self):
        self.sent_messages = {"MessageID": [], "MessageType": [], "MessageTimeLeft" :[], "MessageDelay": [], "MessageRoute" :[]}
        self.received_messages = {"MessageID": [], "MessageType": [], "MessageTimeLeft" :[],"MessageTimeReceived":[], "MessageDelay": [], "MessageRoute" :[],"MessageTarget" : [] }
        self.dummy_messages = {"DroppingNode":[],"DummyID": [], "DummyType": [], "DummyTimeLeft" :[], "DummyDelay": [], "DummyRoute" :[], "DummyPr":[]}
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