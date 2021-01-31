import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import copy
import random
import logging

from include.science_utils import SignalUtils as su
from include.science_utils import TransceiverCharacterization as tc
from include.parameters import Parameters as const
from include.parameters import SigConstants as sc
from include.parameters import ConnectionStatus as cs


class SignalInformation:
    def __init__(self, path):
        self._signal_power = 0
        self._path = path
        self._noise_power = 0
        self._latency = 0
        self._isnr = 0

    def signal_power_increment(self, increment):
        self.signal_power += increment

    def signal_noise_increment(self, increment):
        self.noise_power += increment

    def latency_increment(self, increment):
        self.latency += increment

    def isnr_increment(self, noise):
        self.isnr += noise / self._signal_power

    def cross_node(self):
        self.path = self.path[1:]

    # Getter and setters
    @property
    def signal_power(self):
        return self._signal_power

    @property
    def noise_power(self):
        return self._noise_power

    @property
    def latency(self):
        return self._latency

    @property
    def path(self):
        return self._path

    @property
    def isnr(self):
        return self._isnr

    @signal_power.setter
    def signal_power(self, signal_power):
        if signal_power >= 0:
            self._signal_power = signal_power

    @noise_power.setter
    def noise_power(self, noise_power):
        if noise_power >= 0:
            self._noise_power = noise_power

    @latency.setter
    def latency(self, latency):
        if latency is None or latency >= 0:
            self._latency = latency

    @path.setter
    def path(self, path):
        self._path = path

    @isnr.setter
    def isnr(self, isnr):
        self._isnr = isnr


class Lightpath(SignalInformation):
    def __init__(self, path, channel=None, Rs=0, df=0):
        if channel is not None:
            self._channel = channel
        self._Rs = sc.Rs if Rs == 0 else Rs
        self._df = sc.df if df == 0 else df
        super(Lightpath, self).__init__(path)

    @property
    def channel(self):
        return self._channel

    @property
    def Rs(self):
        return self._Rs

    @property
    def df(self):
        return self._df


class Connection:
    def __init__(self, input, output):
        self._input = input
        self._output = output
        self._signal_power = 0
        self._latency = 0
        self._snr = 0
        self._bit_rate = 0
        self._status = cs.PENDING

    @property
    def status(self):
        return self._status

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

    @property
    def bit_rate(self):
        return self._bit_rate

    @status.setter
    def status(self, status):
        self._status = status

    @latency.setter
    def latency(self, latency):
        if latency is None or latency >= 0:
            self._latency = latency

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @bit_rate.setter
    def bit_rate(self, bit_rate):
        self._bit_rate = bit_rate


class Node:
    def __init__(self, label, input_dict):
        self._label = label
        self._position = input_dict["position"]
        self._connected_nodes = input_dict["connected_nodes"]
        if "transceiver" in input_dict and \
                (input_dict["transceiver"] == const.FLEX_RATE_TRANS or input_dict["transceiver"] == const.SHANNON_TRANS):
            self._transceiver = input_dict["transceiver"]
        else:
            self._transceiver = const.FIXED_RATE_TRANS
        self._switching_matrix = {}
        self._successive = {}

    def propagate(self, signal, previous_node=None):
        path = signal.path
        # If the length of the path is 1 we reached the destination node
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
            # Set the optimized launch power
            signal.signal_power = line.optimized_launch_power(signal.Rs, signal.df)
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

    @property
    def successive(self):
        return self._successive

    @property
    def transceiver(self):
        return self._transceiver

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @connected_nodes.setter
    def connected_nodes(self, connected_nodes):
        self._connected_nodes = connected_nodes

    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix

    @transceiver.setter
    def transceiver(self, transceiver):
        self.transceiver = transceiver


class Line:
    def __init__(self, label, length, gain=None, noise_figure=None):
        self._label = label
        self._length = length
        # In order to take track only the in-line amplifiers
        # the result of the division is rounded up and subtracted by 1
        # if for example the fiber is 240KM and we want an amplifier every 80KM
        # we will have 240KM/80KM = 3-1 = 2 in-line amplifiers
        # one at KM80 and the other at KM160
        self._n_amplifiers = np.ceil(self.length / sc.KmPerA) - 1
        self._gain = sc.AdefGain if gain is None else gain
        self._noise_figure = sc.NF if noise_figure is None else noise_figure
        self._successive = {}
        self._free = [1] * const.N_CHANNELS

    def latency_generation(self):
        return su.latency(self.length)

    def noise_generation(self, signal_power, Rs, df):
        return self.ase_generation() + self.nli_generation(signal_power, Rs, df)

    def ase_generation(self):
        # n_amplifiers is the number of in-line amplifiers which are in the line
        # there is also the need to consider in the ASE calculation the booster and the preamp, so other 2 amplifiers
        return su.ase_noise(self.gain, self.noise_figure, self.n_amplifiers+2)

    def nli_generation(self, signal_power, Rs, df):
        eta_nli = su.eta_nli(Rs, df)
        # The number of fiber spans is equal to the number of inline amplifiers + 1
        return su.nli_noise(eta_nli, signal_power, self.n_amplifiers+1)

    def propagate(self, signal):
        node = self.successive[signal.path[0]]
        # work out the latency and the noise introduced by the current line
        signal.latency_increment(self.latency_generation())
        noise = self.noise_generation(signal.signal_power,
                                      signal.Rs,
                                      signal.df)
        signal.signal_noise_increment(noise)
        signal.isnr_increment(noise)
        # set the channel as occupied if the passed object is a lighpath
        if hasattr(signal, "channel"):
            self.free[signal.channel-1] = 0
        # propagate the signal on the node on the other side of the line
        node.propagate(signal, previous_node=self.label[0])
        return signal

    def optimized_launch_power(self, Rs, df):
        eta_nli = su.eta_nli(Rs, df)
        # The number of fiber spans is equal to the number of inline amplifiers + 1
        return su.optimal_launch_power(eta_nli, self.ase_generation(), self.n_amplifiers+1)

    def spectral_congestion(self):
        return 1 - (np.sum(self._free) / const.N_CHANNELS)

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @property
    def free(self):
        return self._free

    @property
    def gain(self):
        return self._gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @label.setter
    def label(self, label):
        self._label = label

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
            self._nodes[node] = Node(node, current_node)
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
        # The path with order to is coincident with the lines
        all_paths[0] = list(lines)
        # check if there is a direct link and add it to the available paths
        if source+dest in lines:
            found_path.append(source+dest)
        # Collect all the paths which doesn't pass trough the same node twice
        # starting from the all the paths with a grade less we take all the lines which starts from the
        # ending  of the considered path, and which doesn't go to a node already present in the path
        for i in range(len(self.nodes)-2):
            for path in all_paths[i]:
                all_paths[i + 1] += [path + line[1] for line in lines if ((path[-1] == line[0]) & (line[1] not in path))]
        for paths in all_paths[1:]:
            for path in paths:
                if path[0] == source and path[-1] == dest:
                    found_path.append(path)
        return found_path

    # fill the value of snr and latency in a list of Connection objects
    def stream(self, connections, filter_by_snr=False, reset_network=True):
        for conn in connections:
            if filter_by_snr:
                found_path = self.find_best_snr(conn.input, conn.output)
            else:
                found_path = self.find_best_latency(conn.input, conn.output)

            # check if there are available paths
            source_node = self.nodes[conn.input]
            # if the bit rate is equal to 0 the path could be not available or the
            # we didn't reach the minimum GSNR requirement
            if found_path is None:
                self.__abort_connection(conn, cs.BLOCKING_EVENT)
                continue

            # Manually unpack the result in order to handle the None case
            channel = found_path[1]
            found_path = found_path[0]

            found_path_str = found_path.replace("->", "")

            # Start the signal propagation and lock the lines
            sig = Lightpath(found_path_str, channel)
            bit_rate = self.calculate_bit_rate(sig, source_node.transceiver)
            # Check if we don't reach the minimum SNR
            if bit_rate == 0:
                self.__abort_connection(conn, cs.LOW_SNR)
                continue
            sig = self.propagate(sig)
            logging.info(f"DEPLOYED: {found_path} Ch. {channel}")

            # Build a new route space datastructure
            self.__update_route_space()

            # Update the values in connection
            conn.latency = sig.latency
            conn.snr = su.snr_db(sig.isnr)
            conn.bit_rate = bit_rate
            conn.status = cs.DEPLOYED
        # Reset the network occupation
        if reset_network:
            self.reset_network_occupacy()

    def deploy_traffic_matrix(self, matrix, filter_by_snr=False, reset_network=False):
        connections = []
        while len(matrix) > 0:
            n1 = random.choice(list(matrix.keys()))
            n2 = random.choice(list(matrix[n1].keys()))
            c = Connection(n1, n2)
            # Deploy the traffic
            self.stream([c], filter_by_snr=filter_by_snr, reset_network=False)
            connections.append(c)
            if c.status == cs.DEPLOYED:
                # Check if the deployed traffic satisfy the request
                matrix[n1][n2] -= c.bit_rate
                # Remove the elements if we reached the maximum
                if matrix[n1][n2] <= 0:
                    matrix[n1].pop(n2)
            else:
                # Remove the element if it was not possible to deploy the connection
                matrix[n1].pop(n2)
            # Check if deployed all the connections starting from n1
            if len(matrix[n1]) == 0:
                matrix.pop(n1)

        if reset_network:
            self.reset_network_occupacy()

        return connections

    # Draw a graphical representation of the network
    def draw(self, show=True):
        plt.figure()
        G = nx.DiGraph()
        color_map = mpl.colors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"], 10, 1)

        G.add_nodes_from(self.nodes.keys())
        pos = {n: tuple(self.nodes[n].position) for n in self.nodes}
        edges1 = {}
        edges2 = {}
        for line in self.lines:
            if line not in edges2:
                edges1[line] = self.lines[line].spectral_congestion()
                inv_line = line[-1]+line[0]
                edges2[inv_line] = self.lines[inv_line].spectral_congestion()

        # Show bar diagram
        plt.subplot(2, 1, 1)
        plt.bar(list(edges1.keys()) + list(edges2.keys()), list(edges1.values()) + list(edges2.values()))

        # Show first network
        plt.subplot(2, 2, 3)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)
        drawn_edges = nx.draw_networkx_edges(G, pos, edges1.keys(), edge_color=edges1.values(), edge_cmap=color_map, edge_vmin=0, edge_vmax=1)
        pc = mpl.collections.PatchCollection(drawn_edges, cmap=color_map)
        plt.colorbar(pc)

        # Show second network
        plt.subplot(2, 2, 4)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edges2.keys(), edge_color=edges2.values(), edge_cmap=color_map, edge_vmin=0, edge_vmax=1)
        plt.colorbar(pc)

        if show:
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
                        sig = Lightpath(path)
                        sig_info = self.propagate(sig)
                        curr_path = path_string[:-2]
                        weighted_paths.append({
                            "paths": curr_path,
                            "latency": sig_info.latency,
                            "noise": sig_info.noise_power,
                            "snr": su.snr(sig_info.isnr)
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

    # Work out the bit rate of a specific path
    def calculate_bit_rate(self, lightpath, strategy):
        # Get the path data from the weighed paths dataframe
        wp = self.weighted_paths
        path = "->".join([p for p in lightpath.path])
        path_data = wp[(wp.paths == path)]
        if path_data.empty:
            return 0
        snr = path_data.snr.values[0]

        # Check the strategy to be used
        if strategy == const.FIXED_RATE_TRANS:
            if snr >= tc.fixed_rate100(sc.BERt, lightpath.Rs, sc.Bn):
                return 100e9
            else:
                return 0
        elif strategy == const.FLEX_RATE_TRANS:
            if snr < (fr0 := tc.flex_rate0(sc.BERt, lightpath.Rs, sc.Bn)):
                return 0
            elif fr0 <= snr < (fr100 := tc.flex_rate100(sc.BERt, lightpath.Rs, sc.Bn)):
                return 100e9
            elif fr100 <= snr < (fr200 := tc.flex_rate200(sc.BERt, lightpath.Rs, sc.Bn)):
                return 200e9
            elif snr >= fr200:
                return 400e9
        elif strategy == const.SHANNON_TRANS:
            return tc.shannon(lightpath.Rs, sc.Bn, snr)
        return 0

    # #
    # Function utils
    # #

    # This method sets the default values for a blocking event in a Connection object
    def __abort_connection(self, conn, reason):
        logging.warning(f"{'BLOCKING EVENT' if reason == cs.BLOCKING_EVENT else 'LOW SNR'}: {conn.input} -> {conn.output}")
        conn.latency = None
        conn.snr = 0
        conn.bit_rate = 0
        conn.status = reason

    # This method perform the logical AND operation between the switching matrix of the traversed switching nodes
    # and the availability matrix of the traversed lines
    def __return_path_availability(self, path):
        availability = [1 for i in range(const.N_CHANNELS)]
        path_len = len(path)
        for i in range(path_len - 1):
            # Perform the logical and with the switching matrix of only the switching nodes
            # source and destination should not be considered
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
            # return the availability of the channels for the given path
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
