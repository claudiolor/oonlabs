import pandas as pd
import matplotlib.pyplot as plt
import random
from include import virtualnet as vn

net = vn.Network("./resources/nodes_full.json")
net.connect()
connection_data = pd.DataFrame(columns=["path", "snr", "latency"])
connections = []

nodes = list(net.nodes.keys())

for i in range(100):
    n1 = random.choice(nodes)
    n2 = random.choice(nodes)
    while n1 == n2:
        n2 = random.choice(nodes)
    connections.append(vn.Connection(n1, n2, 1))

net.stream(connections[:50], False)
net.stream(connections[50:], True)
tot_snr = 0
tot_lat = 0
for conn in connections:
    if conn.latency is not None:
        tot_snr += conn.snr
        tot_lat += conn.latency
        connection_data = connection_data.append({"path": f"{conn.input}->{conn.output}",
                                                  "snr": conn.snr,
                                                  "latency": conn.latency}, ignore_index=True)
print(f"Avg(snr): {tot_snr/len(connection_data)}")
print(f"Avg(lat): {tot_lat/len(connection_data)}")
print(f"N connections: {len(connection_data)}")
plt.subplot(121)
plt.xticks([])
plt.plot(connection_data.path, connection_data.snr, "go")
plt.subplot(122)
plt.xticks([])
plt.plot(connection_data.path, connection_data.latency, "ro")
plt.show()