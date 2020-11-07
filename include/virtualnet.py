import json
import numpy as np
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


class Network:
    def __init__(self, input):
        self._nodes = {}
        self._lines = {}

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

    def propagate(self, signal):
        # propagate the signal starting from the first node of the path
        first_node = self.nodes[signal.path[0]]
        propagated_signal_info = first_node.propagate(signal)
        return propagated_signal_info

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
                for line in lines:
                    if path[-1] == line[0] and line[1] not in path:
                        new_path = path+line[1]
                        all_paths[i+1].append(new_path)
                        if new_path[0] == source and new_path[-1] == dest:
                            found_path.append(new_path)
        return found_path

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

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines







