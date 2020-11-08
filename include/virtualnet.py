import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SignalInformation:
    def __init__(self, signal_power, path):
        self._signal_power = signal_power
        self._path = path
        self._noise_power = 0
        self._latency = 0

    def signal_power_increment(self, increment):
        self.signal_power += increment

    def signal_noise_increment(self, increment):
        self.noise_power += increment

    def latency_increment(self, increment):
        self.latency += increment

    def cross_node(self):
        self.path = self.path[1:]

    # Getter and setters
    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, signal_power):
        if signal_power >= 0:
            self._signal_power = signal_power

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise_power):
        if noise_power >= 0:
            self._noise_power = noise_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        if latency >= 0:
            self._latency = latency

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path


class Connection:
    def __init__(self, input, output, signal_power, ):
        self._input = input
        self._output = output
        self._signal_power = signal_power
        self._latency = 0
        self._snr = 0

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @property
    def snr(self):
        return self._snr

    @latency.setter
    def latency(self, latency):
        if latency >= 0:
            self._latency = latency

    @snr.setter
    def snr(self, snr):
        self._snr = snr


class Node:
    def __init__(self, label, input_dict):
        self._label = label
        self._position = input_dict["position"]
        self._connected_nodes = input_dict["connected_nodes"]
        self._successive = {}

    def propagate(self, signal):
        path = signal.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            # remove the node from the path
            signal.cross_node()
            # propagate the signal to the right line
            line.propagate(signal)
        return signal

    # Getters and Setters
    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @connected_nodes.setter
    def connected_nodes(self, connected_nodes):
        self._connected_nodes = connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive


class Line:
    def __init__(self, label, length):
        self._label = label
        self._length = length
        self._successive = {}
        self._free = True

    def latency_generation(self):
        LIGHTSPEED2_3 = 199861638.67
        return self.length / LIGHTSPEED2_3

    def noise_generation(self, signal_power):
        return 1e-3 * signal_power * self._length

    def propagate(self, signal):
        node = self.successive[signal.path[0]]
        # work out the latency and the noise introduced by the current line
        signal.latency_increment(self.latency_generation())
        signal.signal_noise_increment(self.noise_generation(signal.signal_power))
        # propagate the signal on the node on the other side of the line
        node.propagate(signal)
        return signal

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @property
    def free(self):
        return self._free

    @free.setter
    def free(self, free):
        self._free = free


class Network:
    def __init__(self, input):
        self._nodes = {}
        self._lines = {}
        self._weighted_paths = pd.DataFrame()

        with open(input, "r") as jsonInput:
            loaded_nodes = json.load(jsonInput)

        for node in loaded_nodes:
            current_node = loaded_nodes[node]
            # create and append the new node in the dictionary
            self._nodes[node] = Node(node, current_node)
            # create a link for each adiacency
            for n in current_node["connected_nodes"]:
                neigh = loaded_nodes[n]
                # workout the distance between the two points
                np_curpos = np.array(current_node["position"])
                np_neighpos = np.array(neigh["position"])
                dist = np.sqrt(np.sum((np_curpos - np_neighpos)**2))
                self._lines[node+n] = Line(node+n, dist)

    # Assign to each network elements a dictionary of the nodes and lines of the network
    # dictionary of nodes to the lines and dictionary of lines to the nodes
    def connect(self):
        for node in self.nodes:
            current_node = self.nodes[node]
            for neigh in current_node.connected_nodes:
                line_label = node + neigh
                current_node.successive[line_label] = self.lines[line_label]
                self.lines[line_label].successive[node] = current_node
                self.lines[line_label].successive[neigh] = self.nodes[neigh]

    # Start the propagation of the signal
    def propagate(self, signal):
        # propagate the signal starting from the first node of the path
        first_node = self.nodes[signal.path[0]]
        propagated_signal_info = first_node.propagate(signal)
        return propagated_signal_info

    # Find all the paths between source and destination node
    def find_paths(self, source, dest):
        all_paths = [[] for i in range(len(self.nodes)-1)]
        found_path = []
        lines = self.lines.keys()
        all_paths[0] = list(lines)
        # check if there is a direct link and add it to the available paths
        if source+dest in lines:
            found_path.append(source+dest)
        for i in range(len(self.nodes)-2):
            for path in all_paths[i]:
                all_paths[i + 1] += [path + line[1] for line in lines if ((path[-1] == line[0]) & (line[1] not in path))]
        for paths in all_paths[1:]:
            for path in paths:
                if path[0] == source and path[-1] == dest:
                    found_path.append(path)
        return found_path

    # fill the value of snr and latency in a list of Connection objects
    def stream(self, connections, filter_by_snr=False):
        lines = self.lines
        for conn in connections:
            df = pd.DataFrame(columns=["path", "snr", "latency"])
            available_paths = self.find_paths(conn.input, conn.output)
            if len(available_paths) == 0:
                conn.latency = 0
                conn.snr = None
                continue

            for path in available_paths:
                # check if the path is free
                free_path = True
                for i in range(len(path)-1):
                    line_label = path[i]+path[i+1]
                    if not lines[line_label].free:
                        free_path = False
                        break
                # don't add the path in the available list if the path is not free
                if not free_path:
                    continue

                sig = SignalInformation(conn.signal_power, path)
                sig = self.propagate(sig)

                df = df.append({"path": path,
                           "snr": 10 * np.log10(sig.signal_power / sig.noise_power),
                           'latency': sig.latency}, ignore_index=True)

            # check if there are available paths
            if df.empty:
                conn.latency = 0
                conn.snr = None
                continue

            if filter_by_snr:
                best_path = df[df.snr == df.snr.max()]
            else:
                # otherwise filter by latency
                best_path = df[df.latency == df.latency.min()]

            # Set the lines in the path as occupied
            path = best_path.path.values[0]
            for i in range(len(path)-1):
                line_label = path[i] + path[i + 1]
                lines[line_label].free = False

            conn.latency = best_path.latency.values[0]
            conn.snr = best_path.snr.values[0]

    # Draw a graphical representation of the network
    def draw(self):
        for node in self.nodes:
            current_node = self.nodes[node]
            pos = current_node.position
            plt.plot(pos[0], pos[1], "bo", markersize=10)
            plt.text(pos[0], pos[1]+30000, node)
            for n in current_node.connected_nodes:
                npos = self.nodes[n].position
                plt.plot([pos[0], npos[0]], [pos[1], npos[1]], "g")
        plt.show()

    # Generate a weighted graph of the network
    def generate_weighted_paths(self):
        nodes = self.nodes
        found_paths = []
        latencies = []
        noises = []
        snrs = []

        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    paths = self.find_paths(node1, node2)
                    for path in paths:
                        path_string = ""
                        for node in path:
                            path_string += node + "->"
                        found_paths.append(path_string[:-2])
                        # Create a new signal
                        sig = SignalInformation(1e-3, path)
                        sig_info = self.propagate(sig)
                        latencies.append(sig_info.latency)
                        noises.append(sig_info.noise_power)
                        snr = 10 * np.log10(sig_info.signal_power / sig_info.noise_power)
                        snrs.append(snr)
        self._weighted_paths["paths"] = found_paths
        self._weighted_paths["latency"] = latencies
        self._weighted_paths["noise"] = noises
        self._weighted_paths["snr"] = snrs

    # find the path with best snr in the precalculated weighted graph
    def find_best_snr(self, source, dest):
        wp = self.weighted_paths
        if wp.empty:
            self.generate_weighted_paths()
        paths = wp[(wp.paths.str[0] == source) & (wp.paths.str[-1] == dest)]
        if paths.empty:
            return None

        # sort the paths by snr in order to be able to take the first available path
        paths = paths.sort_values(by=["snr"], ascending=False)
        av_paths = paths.paths.values
        lines = self._lines

        found = False
        count = 0
        for path in av_paths:
            current_path = path.replace("->", "")
            free_path = True
            # check if the path is free
            for i in range(len(current_path)-1):
                line_label = current_path[i] + current_path[i + 1]
                if not lines[line_label].free:
                    free_path = False
                    break

            if free_path:
                # stop searching since paths are sorted
                found = True
                break
            count += 1

        if found:
            return av_paths[count]
        return None

    # find the path with best snr in the precalculated weighted graph
    def find_best_latency(self, source, dest):
        wp = self.weighted_paths
        if wp.empty:
            self.generate_weighted_paths()
        paths = wp[(wp.paths.str[0] == source) & (wp.paths.str[-1] == dest)]
        if paths.empty:
            return None

        # sort by latency in order to be able to take the first free path
        paths = paths.sort_values(by=["latency"])
        av_paths = paths.paths.values
        lines = self._lines

        found = False
        count = 0
        for path in av_paths:
            current_path = path.replace("->", "")
            free_path = True
            # check if the path is free
            for i in range(len(current_path)-1):
                line_label = current_path[i] + current_path[i + 1]
                if not lines[line_label].free:
                    free_path = False
                    break

            if free_path:
                # stop searching since path are sorted
                found = True
                break
            count += 1
        if found:
            return av_paths[count]
        return None

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_paths(self):
        return self._weighted_paths







