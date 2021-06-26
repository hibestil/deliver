class Job:
    job_id = None  # Order id
    location_index = None  # Index of the order location
    delivery = None  # The amount of carboy that will be delivered in this job
    service = None  # Service duration, in seconds

    def __init__(self, id, location_index, delivery, service):
        self.job_id = id
        self.location_index = location_index
        self.delivery = delivery
        self.service = service
