import os
import argparse
import sys
from pycyclone.io import load_placement
import pythunder
from archipelago.io import load_routing_result
from archipelago.pnr_graph import construct_graph

PE_DELAY = 700
MEM_DELAY = 800
SB_UP_DELAY = 90
SB_DOWN_DELAY = 190
SB_HORIZONTAL_DELAY = 140
RMUX_DELAY = 0
GLB_DELAY = 1100

class PathComponents:
    def __init__(self, glbs = 0, hhops = 0, uhops = 0, dhops = 0, pes = 0, mems = 0, used_regs = 0, available_regs = 0, parent = None):
        self.glbs = glbs
        self.hhops = hhops
        self.uhops = uhops
        self.dhops = dhops
        self.pes = pes
        self.mems = mems
        self.available_regs = available_regs
        self.parent = parent

    def get_total(self):
        total = 0
        total += self.glbs * GLB_DELAY
        total += self.hhops * SB_HORIZONTAL_DELAY
        total += self.uhops * SB_UP_DELAY
        total += self.dhops * SB_DOWN_DELAY
        total += self.pes * PE_DELAY
        total += self.mems * MEM_DELAY
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

    node_to_timing = {node:timing_info[node].get_total() for node in graph.nodes}
    node_to_timing = dict(sorted(reversed(list(node_to_timing.items())), key=lambda item: item[1], reverse=True))
    max_node = list(node_to_timing.keys())[0]
    max_delay = list(node_to_timing.values())[0]

    clock_speed = 1.0e12 / max_delay / 1e6

    print("\nCritical Path Info:")
    print("\tMaximum clock frequency:", clock_speed, "MHz")
    print("\tCritical Path:", max_delay, "ns")
    print(f"\t{max_node}", "glb:", timing_info[max_node].glbs, "horiz hops:",  timing_info[max_node].hhops, "up hops:",  timing_info[max_node].uhops, "down hops:",  timing_info[max_node].dhops, "pes:", timing_info[max_node].pes, "mems:", timing_info[max_node].mems, "\n")

def load_id_to_name(packed_filename):
    f = open(packed_filename, "r")
    lines = f.readlines()
 
    id_to_name = {}
    id_to_name_read = False

    for line in lines:
        if "ID to Names:" in line:
            id_to_name_read = True
        elif "Netlist Bus:" in line:
            id_to_name_read = False
        elif id_to_name_read:
            if len(line.split(":")) > 1:
                id = line.split(":")[0]
                name = line.split(":")[1]
                id_to_name[id] = name.strip()
              
    return id_to_name 

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
    packed_file, placement_file, routing_file = parse_args()

    id_to_name = load_id_to_name(packed_file)
    placement = load_placement(placement_file)
    routing = load_routing_result(routing_file)

    graph = construct_graph(placement, routing, id_to_name)
    sta(graph)
   
main()
