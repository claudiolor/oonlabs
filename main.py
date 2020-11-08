import pandas as pd
import matplotlib.pyplot as plt
import random
from include import virtualnet as vn

net = vn.Network("./resources/node.json")
net.connect()
connection_data = pd.DataFrame(columns=["path", "snr", "latency"])
connections = []
nodes = list(net.nodes.keys())

for i in range(100):
    connections.append(vn.Connection(random.choice(nodes), random.choice(nodes), 1))

net.stream(connections[:50], False)
net.stream(connections[50:], True)

for conn in connections:
    if conn.snr is not None:
        connection_data = connection_data.append({"path": f"{conn.input}->{conn.output}",
                                                  "snr": conn.snr,
                                                  "latency": conn.latency}, ignore_index=True)
plt.subplot(121)
plt.xticks([])
plt.plot(connection_data.path, connection_data.snr, "go")
plt.subplot(122)
plt.xticks([])
plt.plot(connection_data.path, connection_data.latency, "ro")
plt.show()