import socket
from utils import config

# In netutils.py
def recvall(sock, length):
    """
    Receive exactly 'length' bytes from the socket.
    """
    data = b""
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data
