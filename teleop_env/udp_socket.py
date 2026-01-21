import socket
import sys

def create_udp_listener(port=9000):
    """Create a UDP socket that listens on the specified port and prints received data."""
    try:
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Bind the socket to the port
        sock.bind(('0.0.0.0', port))
        
        print(f"UDP listener started on port {port}")
        print("Waiting for messages... (Press Ctrl+C to stop)")
        
        # Listen for incoming messages
        while True:
            data, addr = sock.recvfrom(1024)
            print(f"Received from {addr}: {data.decode('utf-8', errors='ignore')}")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        sock.close()

if __name__ == "__main__":
    create_udp_listener(9000)
