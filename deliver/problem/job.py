class Job:
    """A class used to represent a job object"""
    job_id = None  # Order id
    location_index = None  # Index of the order location
    delivery = None  # The amount of carboy that will be delivered in this job
    service = None  # Service duration, in seconds

    def __init__(self, id, location_index, delivery, service):
        self.job_id = id
        self.location_index = location_index
        self.delivery = delivery
        self.service = service

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str = "[Job] :"
        str += "\tid:{}".format(self.job_id)
        str += "\tlocation_index:{}".format(self.location_index)
        str += "\tdelivery:{}".format(self.delivery)
        str += "\tservice:{}".format(self.service)
        return str
