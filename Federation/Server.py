import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(server_address="0.0.0.0:5588", config={"num_rounds": 3})

