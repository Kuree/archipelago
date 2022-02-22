import os
import json
import argparse
import sys
from pycyclone.io import load_placement
import pythunder
from archipelago.io import load_routing_result
from archipelago.pnr_graph import construct_graph


class PathComponents:
    def __init__(self, glbs=0, hhops=0, uhops=0, dhops=0, pes=0, mems=0,
                 available_regs=0, parent=None):
        self.glbs = glbs
        self.hhops = hhops
        self.uhops = uhops
        self.dhops = dhops
        self.pes = pes
        self.mems = mems
        self.available_regs = available_regs
        self.parent = parent
        self.delays = json.load(open(os.path.dirname(os.path.realpath(__file__)) + "/sta_delays.json"))
        
    def get_total(self):
        total = 0
        total += self.glbs * self.delays['glb']
        total += self.hhops * self.delays['sb_horiz']
        total += self.uhops * self.delays['sb_up']
        total += self.dhops * self.delays['sb_down']
        total += self.pes * self.delays['pe']
        total += self.mems * self.delays['mem']
        return total


def sta(graph):
    nodes = graph.topological_sort()
    timing_info = {}

    for node in nodes:
        comp = PathComponents()
        components = [comp]

        if len(graph.sources[node]) == 0:
            comp = PathComponents()
            comp.glbs = 1
            components.append(comp)

        for parent in graph.sources[node]:

            comp = PathComponents()

            if parent in timing_info:
                comp.glbs = timing_info[parent].glbs
                comp.hhops = timing_info[parent].hhops
                comp.uhops = timing_info[parent].uhops
                comp.dhops = timing_info[parent].dhops
                comp.pes = timing_info[parent].pes
                comp.mems = timing_info[parent].mems
                comp.available_regs = timing_info[parent].available_regs
                comp.parent = parent

            g_node = graph.get_node(node)
            if g_node.type_ == "tile":
                if node[0] == 'p':
                    comp.pes += 1
                elif node[0] == 'm':
                    comp.mems += 1
                elif node[0] == 'I' or node[0] == 'i':
                    comp.glbs += 1
            else:
                if g_node.route_type == "SB":
                    if g_node.io == 1:
                        if g_node.side == 3:
                            comp.uhops += 1
                        elif g_node.side == 1:
                            comp.dhops += 1
                        else:
                            comp.hhops += 1
                elif g_node.route_type == "RMUX":
                    if graph.get_node(parent).route_type != "REG":
                        comp.available_regs += 1
                if graph.node_latencies[node] != 0:
                    comp.glbs = 0
                    comp.hhops = 0
                    comp.uhops = 0
                    comp.dhops = 0
                    comp.pes = 0
                    comp.mems = 0
                    comp.available_regs = 0
                    comp.parent = None
            components.append(comp)

        maxt = 0
        max_comp = components[0]
        for comp in components:
            if comp.get_total() > maxt:
                maxt = comp.get_total()
                max_comp = comp

        timing_info[node] = max_comp

    node_to_timing = {
        node: timing_info[node].get_total() for node in graph.nodes}
    node_to_timing = dict(sorted(reversed(
        list(node_to_timing.items())), key=lambda item: item[1], reverse=True))
    max_node = list(node_to_timing.keys())[0]
    max_delay = list(node_to_timing.values())[0]

    clock_speed = 1.0e12 / max_delay / 1e6

    print("\nCritical Path Info:")
    print("\tMaximum clock frequency:", clock_speed, "MHz")
    print("\tCritical Path:", max_delay, "ns")
    print(f"\t{max_node}", "glb:", timing_info[max_node].glbs,
          "horiz hops:", timing_info[max_node].hhops,
          "up hops:",  timing_info[max_node].uhops,
          "down hops:",  timing_info[max_node].dhops,
          "pes:", timing_info[max_node].pes,
          "mems:", timing_info[max_node].mems, "\n")


def parse_args():
    parser = argparse.ArgumentParser("CGRA Retiming tool")
    parser.add_argument("-a", "--app", "-d", required=True,
                        dest="application", type=str)
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
    packed_file, placement_file, routing_file = parse_args()

    id_to_name = pythunder.io.load_id_to_name(packed_file)
    placement = load_placement(placement_file)
    routing = load_routing_result(routing_file)

    graph = construct_graph(placement, routing, id_to_name)
    
    while graph.fix_cycles():
        pass

    sta(graph)


if __name__ == "__main__":
    main()
