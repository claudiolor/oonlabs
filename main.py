import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import logging

from include import virtualnet as vn
from include.parameters import ConnectionStatus as cs


def _parse_arguments():
    argp = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    working_mode = argp.add_mutually_exclusive_group(required=True)
    # Operating modes
    working_mode.add_argument('-g', '--generate', action='store', metavar='G', type=int,
                              help="Generate the specified number of random connections")
    working_mode.add_argument('-f', '--file', metavar='F', action='store',
                              help="Import the connections from the specified file")
    working_mode.add_argument('-m', '--matrix', metavar='M', action='store',
                              help="Generates random connections with a traffic matrix "
                                   "requiring M * 100Gpbs for each possible connection, where M is the input data")
    working_mode.add_argument('-p', '--pair', metavar=('A', 'B', 'N'), action='store', nargs=3,
                              help="Create N connections between nodes A and B.")

    # Options
    argp.add_argument('-o', '--output', action='store', metavar='O',
                      help="If -g, --generate or -m, --matrix are used, the randomly "
                           "generated connections are exported in the specified file")
    argp.add_argument('-v', '--verbose', action='store_true',
                      help="Display some additional information about the paths deployment")
    argp.add_argument('-V', '--really-verbose', action='store_true',
                      help="Display a rich amount of information about the paths deployment")
    argp.add_argument('-l', '--best-latency', action='store_false',
                      help="Select the path by the best latency in the lightpath deployment")

    # Argument
    argp.add_argument('network', metavar='N',
                        help='The json file path representing the network to be imported')

    # Perform parsing
    args = argp.parse_args()

    # Check the current mode
    if args.generate is not None:
        mode = "-g"
    elif args.file is not None:
        mode = "-f"
    elif args.matrix is not None:
        mode = "-m"
    elif args.pair is not None:
        mode = "-p"

    vars(args)["mode"] = mode

    return args


def export_connections(connections):
    out_connections = np.array(connections)
    np.save(args.output, out_connections, allow_pickle=True)


if __name__ == "__main__":
    args = _parse_arguments()

    if args.verbose:
        logging.basicConfig(level=logging.WARNING)
    elif args.really_verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    net = vn.Network(args.network)
    net.connect()
    connections = []
    nodes = list(net.nodes.keys())

    mode = args.mode

    if mode == "-g":
        n_connections = int(args.generate)
        for i in range(n_connections):
            n1 = random.choice(nodes)
            n2 = random.choice(nodes)
            while n1 == n2:
                n2 = random.choice(nodes)
            connections.append(vn.Connection(n1, n2))
        if args.output is not None:
            export_connections(connections)

        net.stream(connections, filter_by_snr=args.best_latency)
    elif mode == "-p":
        n_connection = int(args.pair[2])
        n1 = args.pair[0]
        n2 = args.pair[1]
        if n1 not in nodes or n2 not in nodes:
            logging.error(f"Invalid nodes for connections {n1}->{n2}")
            exit(1)

        for i in range(n_connection):
            connections.append(vn.Connection(n1, n2))

        net.stream(connections, filter_by_snr=args.best_latency)
    elif mode == "-f":
        connections = np.load(args.file, allow_pickle=True).tolist()

        net.stream(connections, filter_by_snr=args.best_latency)
    elif mode == "-m":
        M = int(args.matrix)
        traffic_matrix = {n : {n1: M*100e9 for n1 in nodes if n != n1} for n in nodes}
        connections = net.deploy_traffic_matrix(traffic_matrix, filter_by_snr=args.best_latency)
        if args.output is not None:
            export_connections(connections)

    tot_snr = 0
    tot_lat = 0
    tot_bit_rate = 0
    tot_low_snr = 0
    tot_blocking_events = 0
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
        else:
            if conn.status == cs.LOW_SNR:
                tot_low_snr += 1
            elif conn.status == cs.BLOCKING_EVENT:
                tot_blocking_events += 1

    connection_data = pd.DataFrame(conn_res_list)

    print(f"Avg(snr): {tot_snr/len(connection_data)}")
    print(f"Avg(lat): {tot_lat/len(connection_data)}")
    print(f"Avg(bit_rate): {tot_bit_rate/(len(connection_data)*(1e9))} Gbps")
    print(f"Total capacity: {tot_bit_rate/1e12} Tbps")
    print(f"\nN. blocking events: {tot_blocking_events}")
    print(f"N. too low snr: {tot_low_snr}")
    print(f"N connections: {len(connection_data)}")

    plt.figure()
    plt.subplot(231)
    plt.xticks([])
    plt.xlabel("SNR")
    plt.plot(connection_data.path, connection_data.snr, "go")
    plt.subplot(232)
    plt.xticks([])
    plt.xlabel("Latency")
    plt.plot(connection_data.path, connection_data.latency, "ro")
    plt.subplot(233)
    plt.xticks([])
    plt.xlabel("Bit rate")
    plt.plot(connection_data.path, connection_data.bit_rate, "bo")
    # Histograms
    plt.subplot(234)
    plt.xlabel("SNR")
    plt.hist(connection_data.snr, color='g')
    plt.subplot(235)
    plt.xlabel("Latency")
    plt.hist(connection_data.latency, color='r')
    plt.subplot(236)
    plt.xlabel("Bit rate")
    plt.hist(connection_data.bit_rate, color='b')

    net.draw(show=False)
    plt.show()
