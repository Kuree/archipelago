import os
import argparse
from pycyclone.io import load_placement
from canal.pnr_io import __parse_raw_routing_result
from .pipeline import construct_graph, sta, load_netlist


def sta_analysis(app_dir, placement, routing, id_to_name):
    graph = construct_graph(placement, routing, id_to_name)
    curr_freq, crit_path, crit_nets = sta(graph)

def parse_args():
    parser = argparse.ArgumentParser("CGRA Retiming tool")
    parser.add_argument("-a", "--app", "-d", required=True, dest="application", type=str, help="Application directory")
    args = parser.parse_args()
    dirname = os.path.join(args.application, "bin")
    netlist = os.path.join(dirname, "design.packed")
    assert os.path.exists(netlist), netlist + " does not exist"
    placement = os.path.join(dirname, "design.place")
    assert os.path.exists(placement), placement + " does not exists"
    route = os.path.join(dirname, "design.route")
    assert os.path.exists(route), route + " does not exists"
    return netlist, placement, route


def main():
    netlist_file, placement_file, routing_file = parse_args()

    print("Loading netlist")
    netlist, id_to_name = load_netlist(netlist_file)
    print("Loading placement")
    placement = load_placement(placement_file)
    print("Loading routing")
    routing = __parse_raw_routing_result(routing_file)

    app_dir = os.path.dirname(netlist_file)

    sta_analysis(app_dir, placement, routing, id_to_name)

    
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    main()

