class Customer:
    def __init__(self, cid, service_duration, demand):
        self.id = cid
        self.service_duration = service_duration
        self.demand = demand

    def __str__(self):
        return "[Customer] :: id:{} | service_duration:{} | demand:{}".format(self.id,
                                                                              self.service_duration,
                                                                              self.demand)
