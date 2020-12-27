import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from include import virtualnet as vn


def usage():
    print("Usage: \n "
          "network_json -g n_connections [connections_json_output] \n"
          "- Create n_connections random connection and save them in a file if a path is specified. \n "
          "network_json -f connection_json_input \n"
          "- Open a file which already contains previously generated connections")
    sys.exit(1)


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 3 or argc > 5:
        usage()
    network_json = sys.argv[1]

    net = vn.Network(network_json)
    net.connect()
    connections = []
    nodes = list(net.nodes.keys())

    mode = sys.argv[2]

    if mode == "-g":
        n_connections = int(sys.argv[3])
        for i in range(n_connections):
            n1 = random.choice(nodes)
            n2 = random.choice(nodes)
            while n1 == n2:
                n2 = random.choice(nodes)
            connections.append(vn.Connection(n1, n2))
        if argc == 5:
            out_filename = sys.argv[4]
            out_connections = np.array(connections)
            np.save(out_filename, out_connections, allow_pickle=True)
    elif mode == "-f":
        in_filename = sys.argv[3]
        connections = np.load(in_filename, allow_pickle=True).tolist()
    else:
        usage()

    net.stream(connections, filter_by_snr=True)

    tot_snr = 0
    tot_lat = 0
    tot_bit_rate = 0
    conn_res_list = []
    for conn in connections:
        if conn.latency is not None:
            tot_snr += conn.snr
            tot_lat += conn.latency
            tot_bit_rate += conn.bit_rate
            connection_data = conn_res_list.append({"path": f"{conn.input}->{conn.output}",
                                                    "snr": conn.snr,
                                                    "latency": conn.latency,
                                                    "bit_rate": conn.bit_rate
                                                    })
    connection_data = pd.DataFrame(conn_res_list)

    print(f"Avg(snr): {tot_snr/len(connection_data)}")
    print(f"Avg(lat): {tot_lat/len(connection_data)}")
    print(f"Avg(bit_rate): {tot_bit_rate/len(connection_data)}")
    print(f"N connections: {len(connection_data)}")
    plt.subplot(131)
    plt.xticks([])
    plt.xlabel("SNR")
    plt.plot(connection_data.path, connection_data.snr, "go")
    plt.subplot(132)
    plt.xticks([])
    plt.xlabel("Latency")
    plt.plot(connection_data.path, connection_data.latency, "ro")
    plt.subplot(133)
    plt.xticks([])
    plt.xlabel("Bit rate")
    plt.plot(connection_data.path, connection_data.bit_rate, "bo")
    plt.show()
