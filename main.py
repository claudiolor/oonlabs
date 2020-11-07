import pandas as pd
import numpy as np
from include import virtualnet as vn

df = pd.DataFrame()
foundPaths = []
latencies = []
noises = []
snrs = []

net = vn.Network("./resources/node.json")
net.connect()

for node1 in net.nodes:
    for node2 in net.nodes:
        if node1 != node2:
            paths = net.find_paths(node1, node2)
            for path in paths:
                path_string = ""
                for node in path:
                    path_string += node + "->"
                foundPaths.append(path_string[:-2])
                # Create a new signal
                sig = vn.SignalInformation(1e-3, path)
                sigInfo = net.propagate(sig)
                latencies.append(sigInfo.latency)
                noises.append(sigInfo.noise_power)
                snr = 10 * np.log10(sigInfo.signal_power / sigInfo.noise_power)
                snrs.append(snr)
df["paths"] = foundPaths
df["latency"] = latencies
df["noise"] = noises
df["snr"] = snrs
print(df)
net.draw()
