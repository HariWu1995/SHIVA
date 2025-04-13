import random as rd
import socket


def find_available_ports(count: int = 1, 
                    start_port: int = 1024, 
                      end_port: int = 65535):
    """
    Find a list of available ports on localhost.

    Args:
        count (int): Number of ports to find.
        start_port (int): Starting port in the range to check.
        end_port (int): Ending port in the range to check.

    Returns:
        List[int]: A list of available port numbers.
    """
    available_ports = []
    tried_ports = set()
    for port in range(start_port, end_port):
        if len(available_ports) >= count:
            break
        if port in tried_ports:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                s.listen(1)
                available_ports.append(port)
            except OSError:
                continue
            finally:
                tried_ports.add(port)
    return available_ports

