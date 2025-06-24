import struct
import pickle
import socket, threading
from time import monotonic
from pympler import asizeof

from xmkckks_wrapper import *
import config_FL
import saveCSV

start_time = monotonic()

server = MPHEServer(server_id=0)
"""This class is responsible for receiving the keys from each client."""
class KeyManager:  
    def __init__(self, host, port):
        self.aggregated_key = []
        self.clients = []
        self.lock = threading.Lock()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.server_socket.setsockopt(socket.SOL_SOCKET,  1)
        self.server_socket.bind((host, port))

        self.server_socket.listen()

    def accept_clients(self, num_clients):
        threads = []

        for _ in range(num_clients):
            client_socket, _ = self.server_socket.accept()
            self.clients.append(client_socket)

            # Create a new thread for each client
            thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

    def handle_client(self, client_socket):
        try:
            # Receive pickled key data
            #current_thread = threading.current_thread()
            #print(f"Thread {current_thread.name} handling client: {client_socket.getpeername()}")

            client_key_data = self.recv_msg(client_socket)

            if client_key_data is not None:
                # Deserialize the pickled data using pickle
                client_key = pickle.loads(client_key_data)
                self.add_client_keys(client_key)
            else:
                print("Error receiving client key data.")
        except pickle.UnpicklingError as e:
            print(f"Error unpickling client key data: {e}")
        finally:
            # Close the client socket after handling the key
            pass
            client_socket.close()

    def recv_msg(self, sock):
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            print("Error: Failed to receive message length.")
            return None

        msglen = struct.unpack('>I', raw_msglen)[0]
        print("Received key length:", msglen)

        # Read the message data
        msg_data = self.recvall(sock, msglen)
        #print("Received message data:", msg_data)

        return msg_data

    def recvall(self, sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def add_client_keys(self, client_keys):
        with self.lock:
            client_keys = [client_keys]
            self.aggregated_key += client_keys

    def get_aggregated_key(self):
        with self.lock:
            return self.aggregated_key

    def reset_keys(self):
        self.aggregated_key = []

def send_data_to_clients(data, server_address):
    def send_data():
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # Connect to the server
            client_socket.connect(server_address)
            
            # Serialize data
            data_pickle = pickle.dumps(data)

            # Send the length of the pickled data
            client_socket.sendall(struct.pack('>I', len(data_pickle)))

            # Send the actual pickled data
            client_socket.sendall(data_pickle)

            print(f"Data sent to server at {server_address}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Close the socket
            client_socket.close()

    # Create a thread for sending data
    thread = threading.Thread(target=send_data)
    thread.start()
    thread.join()

class ClientThread(threading.Thread):
        def __init__(self,clientAddress,clientsocket,params):
            threading.Thread.__init__(self)
            self.csocket = clientsocket
            self.params=params
            
        def run(self):
            try:
                data = self.params
                msg = struct.pack('>I', len(data)) + data
                self.csocket.sendall(msg)
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                # Close the client socket after sending the data
                self.csocket.close()
            
            path1 = 'eval_metrics/fl_enc/Params_size_Dealer'
            data_size = asizeof.asizeof(data)
            saveCSV.save(path1, data_size,'data size', config_FL.num_client(), 'nb_client')

M = config_FL.num_client()
LOCALHOST = config_FL.get_localhost()
km_ip, km_port_send, km_port_recv = config_FL.get_key_manager_addr()
key_manager = KeyManager(LOCALHOST, km_port_send)  
key_manager.accept_clients(num_clients=config_FL.num_client())  # the number of clients coming from config_FL
print('Aggregating Keys...')

aggregated_key_list1 = key_manager.get_aggregated_key()
aggregated_key_list= server.aggregate_ringPs(aggregated_key_list1)
"""this is mandatory to keep this sleep for parallel processing"""
#time.sleep(2)   

server_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_soc.bind((LOCALHOST, km_port_recv))
print('Waiting for clients')

i=0
while True:
    server_soc.listen(M)
    clientsock, clientAddress = server_soc.accept()
    params = pickle.dumps(aggregated_key_list)
    newthread = ClientThread(clientAddress, clientsock, params)
    newthread.start()
    i += 1
    if(i==M):       #NOTE Multi Threading is used to take care of all the receipients successful receive 
            #server_soc.close()
            break

path = 'eval_metrics/fl_enc/runtime_Dealer'
runtime = monotonic() - start_time
saveCSV.save(path, runtime,'runtime', config_FL.num_client(), 'nb_client')
server_soc.close()