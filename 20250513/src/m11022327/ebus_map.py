class BusInfo:
    def __init__(self, bus_id):
        self.bus_id = bus_id

    def get_route_info(self):
        """
        Retrieve route information for the bus.
        To be implemented.
        """
        pass

    def get_stop_info(self):
        """
        Retrieve stop information for the bus.
        To be implemented.
        """
        pass

    def get_arrival_time_info(self):
        """
        Retrieve arrival time information for the bus.
        To be implemented.
        """
        pass


if __name__ == "__main__":
    bus = BusInfo("M25305")
    print(f"Bus ID: {bus.bus_id}")
    # Example usage of the methods (to be implemented)
    # bus.get_route_info()
    # bus.get_stop_info()
    # bus.get_arrival_time_info()