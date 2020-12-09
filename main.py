import pandas as pd
import matplotlib.pyplot as plt
import random
from include import virtualnet as vn


if __name__ == "__main__":
    net = vn.Network("./resources/nodes_full_flex_rate.json")
    net.connect()
    connections = []

    nodes = list(net.nodes.keys())

    for i in range(100):
        n1 = random.choice(nodes)
        n2 = random.choice(nodes)
        while n1 == n2:
            n2 = random.choice(nodes)
        connections.append(vn.Connection(n1, n2, 1))

    net.stream(connections, filter_by_snr=False)
    # net.stream(connections[50:], True)
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
    plt.plot(connection_data.path, connection_data.snr, "go")
    plt.subplot(132)
    plt.xticks([])
    plt.plot(connection_data.path, connection_data.latency, "ro")
    plt.subplot(133)
    plt.xticks([])
    plt.plot(connection_data.path, connection_data.bit_rate, "bo")
    plt.show()