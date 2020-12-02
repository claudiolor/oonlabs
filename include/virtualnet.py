import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from include.science_utils import SignalUtils as su
from include.parameters import Parameters as const


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


class Lightpath(SignalInformation):
    def __init__(self, signal_power, path, channel):
        self.channel = channel
        super(Lightpath, self).__init__(signal_power, path)


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
        self._switching_matrix = {}
        self._successive = {}

    def propagate(self, signal, previous_node=None):
        path = signal.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            # remove the node from the path
            signal.cross_node()
            # update switching matrix if we are streaming
            if hasattr(signal, "channel") and previous_node is not None:
                sm_row = self.switching_matrix[previous_node][line_label[1]]
                channel = signal.channel-1
                # update the current channel
                sm_row[channel] = 0
                if channel > 0:
                    sm_row[channel-1] = 0
                if channel < const.N_CHANNELS-1:
                    sm_row[channel+1] = 0
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

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix


class Line:
    def __init__(self, label, length):
        self._label = label
        self._length = length
        self._successive = {}
        self._free = [1] * const.N_CHANNELS

    def latency_generation(self):
        return su.latency(self.length)

    def noise_generation(self, signal_power):
        return su.noise(signal_power, self.length)

    def propagate(self, signal):
        node = self.successive[signal.path[0]]
        # work out the latency and the noise introduced by the current line
        signal.latency_increment(self.latency_generation())
        signal.signal_noise_increment(self.noise_generation(signal.signal_power))
        # set the channel as occupied if the passed object is a lighpath
        if hasattr(signal, "channel"):
            self.free[signal.channel-1] = 0
        # propagate the signal on the node on the other side of the line
        node.propagate(signal, previous_node=self.label[0])
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
        self._weighted_paths = pd.DataFrame(columns=["paths", "latency", "noise", "snr"])
        self._route_space = pd.DataFrame(columns=["paths"]+[f"Ch{i+1}" for i in range(const.N_CHANNELS)])
        self._switching_matrices = {}

        with open(input, "r") as jsonInput:
            loaded_nodes = json.load(jsonInput)

        for node in loaded_nodes:
            current_node = loaded_nodes[node]
            # create and append the new node in the dictionary
            self._nodes[node] = Node(node, current_node);
            self._switching_matrices[node] = current_node["switching_matrix"]
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
            current_node.switching_matrix = copy.deepcopy(self._switching_matrices[node])
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
        for conn in connections:
            if filter_by_snr:
                found_path = self.find_best_snr(conn.input, conn.output)
            else:
                found_path = self.find_best_latency(conn.input, conn.output)

            # check if there are available paths
            if found_path is None:
                conn.latency = None
                conn.snr = 0
                continue
            print(found_path)
            # Manually unpack the result in order to handle the None case
            channel = found_path[1]
            found_path = found_path[0]

            found_path_str = found_path.replace("->", "")

            # Start the signal propagation and lock the lines
            sig = Lightpath(conn.signal_power, found_path_str, channel)
            sig = self.propagate(sig)

            # Build a new route space datastructure
            self.__update_route_space()

            # Update the values in connection
            conn.latency = sig.latency
            conn.snr = su.snr(sig.signal_power, sig.noise_power)
        # Reset the network occupation
        self.reset_network_occupacy()

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
    def generate_data_structures(self):
        nodes = self.nodes
        weighted_paths = []
        route_space = []

        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    paths = self.find_paths(node1, node2)
                    for path in paths:
                        path_string = ""
                        for node in path:
                            path_string += node + "->"
                        # Create a new signal
                        sig = SignalInformation(1e-3, path)
                        sig_info = self.propagate(sig)
                        curr_path = path_string[:-2]
                        weighted_paths.append({
                            "paths": curr_path,
                            "latency": sig_info.latency,
                            "noise": sig_info.noise_power,
                            "snr": su.snr(sig_info.signal_power, sig_info.noise_power)
                        })

                        # Add record in route space for line availability
                        ch_availability = self.__return_path_availability(path)
                        route_space.append({**{
                            "paths": curr_path
                        }, **{
                            f"Ch{i+1}": ch_availability[i] for i in range(const.N_CHANNELS)
                        }})

        # Create the new dataframe from the lists
        self._weighted_paths = pd.DataFrame(weighted_paths)
        self._route_space = pd.DataFrame(route_space)

    # find the path with best snr in the precalculated weighted graph
    def find_best_snr(self, source, dest):
        if self.weighted_paths.empty:
            self.generate_data_structures()
        wp = self.weighted_paths

        paths = wp[(wp.paths.str[0] == source) & (wp.paths.str[-1] == dest)]
        if paths.empty:
            return None

        # sort the paths by snr in order to be able to take the first available path
        paths = paths.sort_values(by=["snr"], ascending=False)
        av_paths = paths.paths.values
        return self.__select_first_available_path(av_paths)

    # find the path with best snr in the precalculated weighted graph
    def find_best_latency(self, source, dest):
        if self.weighted_paths.empty:
            self.generate_data_structures()
        wp = self.weighted_paths

        paths = wp[(wp.paths.str[0] == source) & (wp.paths.str[-1] == dest)]
        if paths.empty:
            return None

        # sort by latency in order to be able to take the first free path
        paths = paths.sort_values(by=["latency"])
        av_paths = paths.paths.values
        return self.__select_first_available_path(av_paths)

    # This function reset the switching matrices and the occupacy of the lines
    def reset_network_occupacy(self):
        # Revert the switching matrix of all the nodes
        for node in self.nodes:
            self.nodes[node].switching_matrix = copy.deepcopy(self._switching_matrices[node])
        # Revert the occupation of the lines
        for line in self.lines:
            self.lines[line].free = [1] * const.N_CHANNELS
        self.__update_route_space()

    # #
    # Function utils
    ##

    # This method perform the logical and operation between the switching matrix of the traversed switching nodes
    # and the availability matrix of the traversed lines
    def __return_path_availability(self, path):
        availability = [1 for i in range(const.N_CHANNELS)]
        path_len = len(path)
        for i in range(path_len - 1):
            if 0 < i < (path_len - 1):
                availability = np.multiply(availability, self.nodes[path[i]].switching_matrix[path[i - 1]][path[i + 1]])
            availability = np.multiply(availability, self.lines[path[i] + path[i + 1]].free)
        return availability

    # This method is used in order to select the first path which is available in a given list
    def __select_first_available_path(self, av_paths):
        count = 0
        channel = 0
        for path in av_paths:
            path_state = self.route_space[(self.route_space.paths == path)]
            # check if the path is free
            for c in range(1, const.N_CHANNELS + 1):
                if path_state[f"Ch{c}"].values[0]:
                    channel = c
                    break

            if channel > 0:
                # stop searching since paths are sorted
                break
            count += 1

        if channel > 0:
            return av_paths[count], channel
        return None

    # This function update the route space according to the current values of switching matrices
    # and availability of the channels of the line
    def __update_route_space(self):
        route_space = []
        rs = self.route_space
        for path in rs.paths:
            path_str = path.replace("->", "")
            ch_availability = self.__return_path_availability(path_str)
            route_space.append({**{
                "paths": path
            }, **{
                f"Ch{i + 1}": ch_availability[i] for i in range(const.N_CHANNELS)
            }})
        self._route_space = pd.DataFrame(route_space)

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def route_space(self):
        return self._route_space







