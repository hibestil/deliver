class Customer:
    def __init__(self, cid, service_duration, demand, location_index):
        self.id = cid
        self.service_duration = service_duration
        self.demand = demand
        self.location_index = location_index

    def __str__(self):
        return "[Customer] :: id:{} | service_duration:{} | demand:{} | index: {}".format(self.id,
                                                                                        self.service_duration,
                                                                                        self.demand,
                                                                                        self.location_index)
