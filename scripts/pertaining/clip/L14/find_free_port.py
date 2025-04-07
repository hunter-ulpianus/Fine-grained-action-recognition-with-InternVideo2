import socket
import os

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    port = find_free_port()
    output_file = os.path.join("/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/scripts/pretraining/clip/L14", "free_port.txt")
    with open(output_file, "w") as f:
        f.write(str(port))
    print(f"Free port {port} written to {output_file}")
