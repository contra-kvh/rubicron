import json
import os
import socket
from typing import Tuple
import numpy as np
import threading
import torch

from utils import config
from utils import net as netutils
from model.builder import RLModel
from model.local_pred import predict_local

class ServerSocket:
    def __init__(self, host: str, port: int):
        """
        The server object listens to connections and creates client handlers
        for every client (multi-threaded).

        It receives inputs from the clients and returns the predictions to the correct client.
        """
        self.host = host
        self.port = port
        # Load PyTorch model
        self.device = config.DEVICE
        self.model = RLModel(input_shape=config.INPUT_SHAPE, output_shape=(4672, 1))
        self.model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, "model.pt"), map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        # Test prediction to warm up model
        test_data = np.random.choice(a=[False, True], size=(1, *config.INPUT_SHAPE), p=[0, 1]).astype(np.bool_)
        with torch.no_grad():
            p, v = predict_local(self.model, test_data)
        del test_data, p, v

    def start(self):
        """
        Start the server and listen for connections.
        """
        print(f"Starting server on {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(24)  # Queue up to 24 requests
        print(f"Server started on {self.sock.getsockname()}")
        try:
            while True:
                self.accept()
                print(f"Current thread count: {threading.active_count()}.")
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"Error: {e}")
            self.sock.close()

    def accept(self):
        """
        Accept a connection and create a client handler for it.
        """
        print("Waiting for client...")
        client, address = self.sock.accept()
        print(f"Client connected from {address}")
        clh = ClientHandler(client, address, self.model)
        clh.start()

    def stop(self):
        print("Stopping server...")
        self.sock.close()
        print("Server stopped.")

class ClientHandler(threading.Thread):
    def __init__(self, sock: socket.socket, address: Tuple[str, int], model: RLModel):
        """
        The ClientHandler object handles a single client connection, sends
        inputs to the server, and returns the server's predictions to the client.
        """
        super().__init__()
        self.BUFFER_SIZE = config.SOCKET_BUFFER_SIZE
        self.sock = sock
        self.address = address
        self.model = model
        self.device = config.DEVICE

    def run(self):
        """Create a new thread"""
        print(f"ClientHandler started.")
        while True:
            data = self.receive()
            if data is None or len(data) == 0:
                self.close()
                break
            data = np.array(np.frombuffer(data, dtype=bool))
            data = data.reshape((1, *config.INPUT_SHAPE))
            # make prediction
            p, v = predict_local(inputs=data, model=self.model)
            p, v = p[0].cpu().numpy().tolist(), float(v[0][0])
            response = json.dumps({"prediction": p, "value": v})
            self.send(f"{len(response):010d}".encode('ascii'))
            self.send(response.encode('ascii'))
            

    def receive(self):
        """
        Receive data from the client.
        """
        data = None
        try:
            data_length = self.sock.recv(10)
            if data_length == b'':
                # this happens if the socket connects and then closes without sending data
                return data
            data_length = int(data_length.decode("ascii"))
            data = netutils.recvall(self.sock, data_length)
            if len(data) != 1216:
                data = None
                raise ValueError("Invalid data length, closing socket")
        except ConnectionResetError:
            print(f"Connection reset by peer. Client IP: {str(self.address[0])}:{str(self.address[1])}")
        except ValueError as e:
            print(e)
        return data

    def send(self, data):
        """
        Send data to the client.
        """
        self.sock.send(data)

    def close(self):
        print("Closing connection...")
        self.sock.close()
        print("Connection closed.")


if __name__ == "__main__":
    s = ServerSocket(config.SOCKET_HOST, config.SOCKET_PORT)
    s.start()
