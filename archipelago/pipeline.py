import sys
import os
import argparse
import re
import itertools
import glob
import json
 
from graphviz import Digraph

from pycyclone.io import load_placement
# parse raw routing result
from canal.pnr_io import __parse_raw_routing_result
from typing import Dict, List, NoReturn, Tuple, Set
from .pnr_graph import Node, Graph, KernelNode, KernelGraph

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
    return netlist, placement, route, args.frequency



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

    net_ids = set()

    curr_node = max_node
    net_ids.add(graph.get_node(curr_node).net_id)
    print("\t",curr_node, timing_info[curr_node].get_total(), "glb:", timing_info[curr_node].glbs, "horiz hops:",  timing_info[curr_node].hhops, "up hops:",  timing_info[curr_node].uhops, "down hops:",  timing_info[curr_node].dhops, "pes:", timing_info[curr_node].pes, "mems:", timing_info[curr_node].mems, "regs:", timing_info[curr_node].available_regs)
    crit_path = []
    crit_path.append((curr_node, timing_info[curr_node].get_total()))
    crit_edges = []
    while(True):
        if (timing_info[curr_node].parent, curr_node) in graph.edges:
            crit_edges.append((timing_info[curr_node].parent, curr_node))
        curr_node = timing_info[curr_node].parent
        if curr_node == None:
            break
        net_ids.add(graph.get_node(curr_node).net_id)
        crit_path.append((curr_node, timing_info[curr_node].get_total()))

    crit_path.reverse()

    print("\tCritical Path Nets:", *net_ids)

    return clock_speed, crit_path, crit_edges

def find_break_idx(graph, crit_path):
    crit_path_adjusted = [abs(c - crit_path[-1][1]/2) for n,c in crit_path]
    break_idx = crit_path_adjusted.index(min(crit_path_adjusted))

    if len(crit_path) < 2:
        raise ValueError("Can't find available register on critical path")

    while True:
        if graph.get_node(crit_path[break_idx + 1][0]).route_type == "RMUX" and graph.get_node(crit_path[break_idx][0]).route_type == "SB":
            return break_idx
        break_idx += 1

        if break_idx + 1 >= len(crit_path):
            break_idx = crit_path_adjusted.index(min(crit_path_adjusted))

            while True:
                if graph.get_node(crit_path[break_idx + 1][0]).route_type == "RMUX" and graph.get_node(crit_path[break_idx][0]).route_type == "SB":
                    return break_idx
                break_idx -= 1

                if break_idx < 0:
                    raise ValueError("Can't find available register on critical path")

def reg_into_route(routes, g_break_node_source, new_reg_route_source):
    for net_id, net in routes.items():
        for route in net:
            for idx, segment in enumerate(route):
                if g_break_node_source.to_route() == segment:
                    route.insert(idx + 1, new_reg_route_source.to_route())
                    return 

def break_crit_path(graph, id_to_name, crit_path, placement, routes):
    break_idx = find_break_idx(graph, crit_path)

    break_node_source = crit_path[break_idx][0]
    break_node_dest = graph.sinks[break_node_source][0]
    g_break_node_source = graph.get_node(break_node_source)
    g_break_node_dest = graph.get_node(break_node_dest)

    assert g_break_node_source.type_ == "route" 
    assert g_break_node_source.route_type == "SB" 
    assert g_break_node_dest.type_ == "route" 
    assert g_break_node_dest.route_type == "RMUX"

    x = g_break_node_source.x
    y = g_break_node_source.y
    track = g_break_node_source.track
    bw = g_break_node_source.bit_width
    net_id = g_break_node_source.net_id
    kernel = g_break_node_source.kernel
    side = g_break_node_source.side
    print("\t\tBreaking net:", net_id, "Kernel:", kernel)
    
    dir_map = {0: "EAST", 1: "SOUTH", 2: "WEST", 3: "NORTH"}

    new_segment = ["REG", f"T{track}_{dir_map[side]}", track, x, y, bw]
    new_reg_route_source = segment_to_node(new_segment, net_id)
    new_reg_route_source.reg = True
    new_reg_route_source.update_tile_id()
    new_reg_route_dest = segment_to_node(new_segment, net_id)
    new_reg_tile = Node("tile", x, y, tile_id=f"r_ADDED{graph.added_regs}", kernel = kernel)
    graph.added_regs += 1
    
    graph.edges.remove((break_node_source, break_node_dest))
    graph.add_node(new_reg_route_source)
    graph.node_latencies[new_reg_route_source.tile_id] = 1
    graph.add_node(new_reg_tile)
    graph.node_latencies[new_reg_tile.tile_id] = 0
    graph.add_node(new_reg_route_dest)
    graph.node_latencies[new_reg_route_dest.tile_id] = 0
    

    graph.add_edge(break_node_source, new_reg_route_source)
    graph.add_edge(new_reg_route_source, new_reg_tile)
    graph.add_edge(new_reg_tile, new_reg_route_dest)
    graph.add_edge(new_reg_route_dest, break_node_dest)

    reg_into_route(routes, g_break_node_source, new_reg_route_source)
    placement[new_reg_tile.tile_id] = (new_reg_tile.x, new_reg_tile.y)
    id_to_name[new_reg_tile.tile_id] = f"pnr_pipelining{graph.added_regs}"

    graph.update_sources_and_sinks()
    graph.update_edge_kernels()

def pipeline_input_ios(graph, num_stages, id_to_name, placement, routing):
    for _ in range(num_stages):
        for io_node in graph.get_input_ios():
            if "reset" in graph.get_node(io_node).kernel:
                continue
            path = []
            curr_node = io_node
            idx = 0
            while curr_node not in graph.get_mems() and len(graph.sinks[curr_node]) < 2:
                assert len(graph.sinks[curr_node]) == 1, len(graph.sinks[curr_node])
                path.append((curr_node, idx))
                curr_node = graph.sinks[curr_node][0]
                idx += 1
            break_crit_path(graph, id_to_name, path, placement, routing)

def break_at(graph, node1, id_to_name, placement, routing):    
    path = []
    curr_node = node1
    kernel = graph.get_node(curr_node).kernel

    while len(graph.sinks[curr_node]) == 1:
        if len(graph.sources[graph.sinks[curr_node][0]]) > 1 or graph.get_node(graph.sinks[curr_node][0]).kernel != kernel:
            break
        curr_node = graph.sinks[curr_node][0]

    idx = 0
    while len(graph.sources[curr_node]) == 1:
        if len(graph.sinks[curr_node]) > 1 or graph.get_node(graph.sources[curr_node][0]).kernel != kernel:
            break
        path.append((curr_node, idx))
        curr_node = graph.sources[curr_node][0]
        if curr_node in graph.get_ponds():
            break
    
    if curr_node in graph.get_ponds():        
        print("\t\tFound pond for branch delay matching", curr_node)
        for source in graph.sources[curr_node]:
            if graph.get_node(source).port == "flush":
                continue
            graph.node_latencies[source] += 1
        return

    if len(path) == 0:
        raise ValueError(f"Cant break at node: {node1}")
    path.reverse()

    ret = []

    for p in path:
        ret.append((p[0], idx))
        idx += 1
    break_crit_path(graph, id_to_name, ret, placement, routing)

def add_delay_to_kernel(graph, kernel, added_delay, id_to_name, placement, routing):
    kernel_output_nodes = graph.get_output_tiles_of_kernel(kernel)
    for node in kernel_output_nodes:
        for _ in range(added_delay):
            break_at(graph, node, id_to_name, placement, routing)

def branch_delay_match_all_nodes(graph, id_to_name, placement, routing):
    nodes = graph.topological_sort()
    node_cycles = {}

    for node in nodes:
        cycles = set()
        g_node = graph.get_node(node)


        if len(graph.sources[node]) == 0:
            cycles = {0}

        for parent in graph.sources[node]:
            g_parent = graph.get_node(parent)
            if parent not in node_cycles:
                c = 0
            else:
                c = node_cycles[parent]
            
            if g_node.type_ == "route":
                if graph.node_latencies[node] != 0:
                    if len(graph.sinks[node]) > 0:
                        child = graph.sinks[node][0]
                        if (child not in id_to_name or "d_reg_" not in id_to_name[child]) and (parent not in graph.get_mems()):
                            if c != None: 
                                c += graph.node_latencies[node]
            if parent not in graph.get_mems():
                cycles.add(c)
        
  
        if (node in graph.get_pes() and len(graph.sources[node]) == 0) or (g_node.type_ == "route" and g_node.port == "flush"):
            cycles = {None}
  
        if None in cycles:
            cycles.remove(None)
        if len(graph.sources[node]) > 1 and len(cycles) > 1:
            print(f"Something went wrong, incorrect node delay: {node} {cycles}")

        if len(cycles) > 0:
            node_cycles[node] = max(cycles)
        else:
            node_cycles[node] = None

def branch_delay_match_within_kernels(graph, id_to_name, placement, routing):
    nodes = graph.topological_sort()
    node_cycles = {}

    for node in nodes:
        cycles = set()
        g_node = graph.get_node(node)

        if g_node.kernel not in node_cycles:
            node_cycles[g_node.kernel] = {}

        if len(graph.sources[node]) == 0:
            cycles = {0}

        for parent in graph.sources[node]:
            g_parent = graph.get_node(parent)
            if parent not in node_cycles[g_node.kernel]:
                c = 0
            else:
                c = node_cycles[g_node.kernel][parent]
            
            if c != None and g_node.type_ == "route" and graph.node_latencies[node] != 0:
                if len(graph.sinks[node]) > 0:
                    child = graph.sinks[node][0]
                    if (child not in id_to_name or "d_reg_" not in id_to_name[child]) and (parent not in graph.get_mems()):
                        c += graph.node_latencies[node]

            if not (g_parent.type_ == "route" and g_parent.port == "flush"):
                cycles.add(c)
        
  
        if (node in graph.get_pes() and len(graph.sources[node]) == 0):
            cycles = {None}
  
        if None in cycles:
            cycles.remove(None)
        if len(graph.sources[node]) > 1 and len(cycles) > 1:
            print(f"\tIncorrect delay within kernel: {g_node.kernel} {node} {cycles}")
            print(f"\tFixing branching delays at: {node} {cycles}") 
            source_cycles = [node_cycles[g_node.kernel][source] for source in graph.sources[node] if node_cycles[g_node.kernel][source] != None]
            max_parent_cycles = max(source_cycles)
            for parent in graph.sources[node]:
                if node_cycles[g_node.kernel][parent] != max_parent_cycles and not (graph.get_node(parent).type_ == "route" and graph.get_node(parent).port == "flush"):
                    for _ in range(max_parent_cycles - node_cycles[g_node.kernel][parent]):
                        break_at(graph, graph.sources[parent][0], id_to_name, placement, routing)

        if len(cycles) > 0:
            node_cycles[g_node.kernel][node] = max(cycles)
        else:
            node_cycles[g_node.kernel][node] = None

    for kernel, node_delays in node_cycles.items():
        if "reset" not in kernel:
            kernel_output_delays = set()
            kernel_output_nodes_and_delays = []
            for node in graph.get_outputs_of_kernel(kernel):
                if node_delays[node] != None:
                    kernel_output_delays.add(node_delays[node])
                    kernel_output_nodes_and_delays.append((node, node_delays[node]))

            if len(kernel_output_delays) > 1:
                print(f"\tIncorrect delay at output of kernel: {kernel} {kernel_output_delays}")
                max_parent_cycles = max(kernel_output_delays)
                for node, delay in kernel_output_nodes_and_delays:
                    if delay != max_parent_cycles:
                        print(f"\tFixing branching delays at: {node} {max_parent_cycles - delay}") 
                        for _ in range(max_parent_cycles - delay):
                            break_at(graph, node, id_to_name, placement, routing)

def branch_delay_match_kernels(kernel_graph, graph, id_to_name, placement, routing):
    nodes = kernel_graph.topological_sort()
    node_cycles = {}

    for node in nodes:
        cycles = set()
        g_node = kernel_graph.get_node(node)


        if len(kernel_graph.sources[node]) == 0:
            if g_node.type_ == "compute":
                cycles = {None}
            else:
                cycles = {0}

        for parent in kernel_graph.sources[node]:
            g_parent = kernel_graph.get_node(parent)
            if parent not in node_cycles:
                c = 0
            else:
                c = node_cycles[parent]
            
            if c is not None:
                c += g_node.latency

            if g_parent.kernel != "reset" and not (g_parent.type_ == "mem" and parent[0] == 'm'):
                cycles.add(c)
        
        if None in cycles:
            cycles.remove(None)
        if len(kernel_graph.sources[node]) > 1 and len(cycles) > 1:
            print(f"\tIncorrect kernel delay: {node} {cycles}")
            
            source_cycles = [node_cycles[source] for source in kernel_graph.sources[node] if node_cycles[source] != None]
            max_cycle = max(source_cycles)
            for source in kernel_graph.sources[node]:
                if node_cycles[source] != None and node_cycles[source] != max_cycle:
                    print(f"\tFixing kernel delays at: {source} {max_cycle - node_cycles[source]}")
                    add_delay_to_kernel(graph, source, max_cycle - node_cycles[source], id_to_name, placement, routing)
        if len(cycles) > 0:
            node_cycles[node] = max(cycles)
        else:
            node_cycles[node] = None


def get_compute_unit_cycles(graph, id_to_name, placement, routing):
    nodes = graph.topological_sort()
    node_cycles = {}

    for node in nodes:
        cycles = set()
        g_node = graph.get_node(node)

        if g_node.kernel not in node_cycles:
            node_cycles[g_node.kernel] = {}

        if len(graph.sources[node]) == 0:
            cycles = {0}

        for parent in graph.sources[node]:
            g_parent = graph.get_node(parent)
            if parent not in node_cycles[g_node.kernel]:
                c = 0
            else:
                c = node_cycles[g_node.kernel][parent]
            
            if g_node.type_ == "route":
                if graph.node_latencies[node] != 0:
                    if len(graph.sinks[node]) > 0:
                        child = graph.sinks[node][0]
                        if (child not in id_to_name or "d_reg_" not in id_to_name[child]) and (parent not in graph.get_mems() or parent in graph.get_roms()):
                            if c != None: 
                                c += graph.node_latencies[node]
            if parent not in graph.get_mems() or parent in graph.get_roms():
                cycles.add(c)
        
  
        if (node in graph.get_pes() and len(graph.sources[node]) == 0) or (g_node.type_ == "route" and g_node.port == "flush"):
            cycles = {None}
  
        if None in cycles:
            cycles.remove(None)
        if len(graph.sources[node]) > 1 and len(cycles) > 1:
            print(f"INCORRECT COMPUTE UNIT CYCLES: {g_node.kernel} {node} {cycles}")
        if len(cycles) > 0:
            node_cycles[g_node.kernel][node] = max(cycles)
        else:
            node_cycles[g_node.kernel][node] = None

    kernel_latencies = {}
    for kernel in node_cycles:
        kernel_cycles = [cyc for cyc in node_cycles[kernel].values() if cyc != None]
        kernel_cycles.append(0)
        kernel_latencies[kernel] = max(kernel_cycles)

    return kernel_latencies


def flush_cycles(graph):
    nodes = graph.topological_sort()
    node_cycles = {}

    for io in graph.get_input_ios():
        if graph.get_node(io).kernel == "io1in_reset":
            break
    assert graph.get_node(io).kernel == "io1in_reset"
    flush_cycles = {}

    for mem in graph.get_mems():
        for curr_node in graph.sources[mem]:
            if graph.get_node(curr_node).port == "flush":
                break
        if graph.get_node(curr_node).port != "flush":
            continue
        
        flush_cycles[mem] = 0
        while curr_node != io:
            g_curr_node = graph.get_node(curr_node)
            if g_curr_node.type_ == "route" and graph.node_latencies[curr_node] != 0:
                flush_cycles[mem] += graph.node_latencies[curr_node]
            curr_node = graph.sources[curr_node][0]

    return flush_cycles

def construct_kernel_graph(graph, new_latencies, flush_latencies):
    kernel_graph = KernelGraph()

    graph.regs = None
    graph.shift_regs = None

    compute_tiles = set()
    for tile in graph.get_pes() + graph.get_regs() + graph.get_ponds():
        if tile not in graph.get_shift_regs():
            compute_tiles.add(tile)

    for source in graph.get_tiles():
        if source in compute_tiles:
            source_id = graph.get_node(source).kernel
            kernel_graph.add_node(KernelNode(kernel = source_id))
            kernel_graph.get_node(source_id).latency = new_latencies[source_id]
            kernel_graph.get_node(source_id).type_ = "compute"
        else:
            source_id = source 
            kernel_graph.add_node(KernelNode(mem_id = source_id))
            kernel_graph.get_node(source_id).type_ = "mem"
            if "reset" in graph.get_node(source).kernel:
                kernel_graph.get_node(source_id).kernel = "reset"
        for dest in graph.get_tiles():
            if dest in compute_tiles:
                dest_id = graph.get_node(dest).kernel
                kernel_graph.add_node(KernelNode(kernel = dest_id))
                kernel_graph.get_node(dest_id).latency = new_latencies[dest_id]
                kernel_graph.get_node(dest_id).type_ = "compute"
            else:
                dest_id = dest 
                kernel_graph.add_node(KernelNode(mem_id = dest_id))
                kernel_graph.get_node(dest_id).type_ = "mem"
                if "reset" in graph.get_node(dest).kernel:
                    kernel_graph.get_node(dest_id).kernel = "reset"

            if source_id != dest_id:
                reachable = False
                visited = set()
                queue = []
                queue.append(source)
                visited.add(source)
                while queue:
                    n = queue.pop()

                    if n == dest and n != source:
                        reachable = True

                    for node in graph.sinks[n]:
                        if node not in visited:
                            if graph.get_node(node).type_ == "tile":
                                if node == dest:
                                    reachable = True
                            else:
                                queue.append(node)
                                visited.add(node)

                if reachable:
                    kernel_graph.add_edge(source_id, dest_id)

    kernel_graph.update_sources_and_sinks()
    kernel_graph.print_graph("kernel_graph")
    return kernel_graph


def calculate_latencies(kernel_graph, kernel_latencies):

    nodes = kernel_graph.topological_sort()
    new_latencies = {}
    flush_latencies = {}

    for node in kernel_graph.nodes:
        if kernel_graph.get_node(node).type_ != "mem":
            new_latencies[node] = kernel_graph.get_node(node).latency
        else:
            flush_latencies[node] = kernel_graph.get_node(node).flush_latency
            
    # Unfortunately exact matches between kernels and memories dont exist, so we have to look them up
    sorted_new_latencies = {}
    for k in sorted(new_latencies, key=len,):
        sorted_new_latencies[k] = new_latencies[k]
    for kernel, lat in kernel_latencies.items():
        if "glb" in kernel:
            continue
        new_lat = lat

        if f"op_{kernel}" in sorted_new_latencies:
            new_lat = sorted_new_latencies[f"op_{kernel}"]
        elif "input" in kernel:
            for f_kernel, lat in sorted_new_latencies.items():
                if "input" in f_kernel:
                    new_lat = sorted_new_latencies[f_kernel]
                    break
        else:
            f_kernel = kernel.split("hcompute_")[1]
            if f_kernel in sorted_new_latencies:
                new_lat = sorted_new_latencies[f_kernel]
            else:
                for f_kernel, lat in sorted_new_latencies.items():
                    if kernel in f_kernel:
                        new_lat = sorted_new_latencies[f_kernel]
                        break
                if kernel not in f_kernel:
                   new_lat = None
        if new_lat != None:
            kernel_latencies[kernel] = new_lat
    return kernel_latencies

def update_kernel_latencies(dir_name, graph, id_to_name, placement, routing):
    
    print("\nBranch delay matching within kernels")
    branch_delay_match_within_kernels(graph, id_to_name, placement, routing)

    compute_latencies = get_compute_unit_cycles(graph, id_to_name, placement, routing)
    flush_latencies = flush_cycles(graph)
    kernel_graph = construct_kernel_graph(graph, compute_latencies, flush_latencies)

    print("\nBranch delay matching kernels")
    branch_delay_match_kernels(kernel_graph, graph, id_to_name, placement, routing)

    print("\nChecking delay matching all nodes")
    branch_delay_match_all_nodes(graph, id_to_name, placement, routing)

    compute_latencies = get_compute_unit_cycles(graph, id_to_name, placement, routing)
    flush_latencies = flush_cycles(graph)
    kernel_graph = construct_kernel_graph(graph, compute_latencies, flush_latencies)

    kernel_latencies_file = glob.glob(f"{dir_name}/*_compute_kernel_latencies.json")[0]
    flush_latencies_file = kernel_latencies_file.replace("compute_kernel_latencies", "flush_latencies")
    pond_latencies_file = kernel_latencies_file.replace("compute_kernel_latencies", "pond_latencies")

    assert os.path.exists(kernel_latencies_file)

    f = open(kernel_latencies_file, "r")
    kernel_latencies = json.load(f)

    kernel_latencies = calculate_latencies(kernel_graph, kernel_latencies)

    flush_latencies = {id_to_name[mem_id]: latency for mem_id, latency in flush_latencies.items()}
    pond_latencies = {}
    for pond_node, latency in graph.node_latencies.items():
        g_pond_node = graph.get_node(pond_node)
        if g_pond_node.port == "data_in_pond":
            pond_latencies[id_to_name[graph.sinks[pond_node][0]]] = latency
    
    for kernel, lat in kernel_latencies.items():
        if "input" in kernel:
             kernel_latencies[kernel] = 0

    fout = open(kernel_latencies_file, "w")
    fout.write(json.dumps(kernel_latencies))

    fout = open(flush_latencies_file, "w")
    fout.write(json.dumps(flush_latencies))

    fout = open(pond_latencies_file, "w")
    fout.write(json.dumps(pond_latencies))

    return kernel_latencies

def segment_node_to_string(node):
    if node[0] == "SB":
        return f"{node[0]} ({node[1]}, {node[2]}, {node[3]}, {node[4]}, {node[5]}, {node[6]})"
    elif node[0] == "PORT":
        return f"{node[0]} {node[1]} ({node[2]}, {node[3]}, {node[4]})"
    elif node[0] == "REG":
        return f"{node[0]} {node[1]} ({node[2]}, {node[3]}, {node[4]}, {node[5]})"
    elif node[0] == "RMUX":
        return f"{node[0]} {node[1]} ({node[2]}, {node[3]}, {node[4]})"

def dump_routing_result(dir_name, routing):

    route_name = os.path.join(dir_name, "design.route")

    fout = open(route_name, "w")

    for net_id, route in routing.items():
        fout.write(f"Net ID: {net_id} Segment Size: {len(route)}\n")
        src = route[0]
        for seg_index, segment in enumerate(route):
            fout.write(f"Segment: {seg_index} Size: {len(segment)}\n")

            for node in segment:
                fout.write(f"{segment_node_to_string(node)}\n")
        fout.write("\n")


def dump_placement_result(dir_name, placement, id_to_name):

    place_name = os.path.join(dir_name, "design.place")
    fout = open(place_name, "w")
    fout.write("Block Name			X	Y		#Block ID\n")
    fout.write("---------------------------\n")

    for tile_id, place in placement.items():
        fout.write(f"{id_to_name[tile_id]}\t\t{place[0]}\t{place[1]}\t\t#{tile_id}\n")

def dump_id_to_name(app_dir, id_to_name):
    id_name = os.path.join(app_dir, "design.id_to_name")
    fout = open(id_name, "w")
    for id_, name in id_to_name.items():
        fout.write(f"{id_}: {name}\n")

def load_id_to_name(id_filename):
    fin = open(id_filename, "r")
    lines = fin.readlines()

    id_to_name = {}

    for line in lines:
        id_to_name[line.split(": ")[0]] = line.split(": ")[1].rstrip()

    return id_to_name
     

def pipeline_pnr(app_dir, placement, routing, id_to_name, netlist, load_only):
    if load_only:
        id_to_name_filename = os.path.join(app_dir, f"design.id_to_name")
        if os.path.isfile(id_to_name_filename):
            id_to_name = load_id_to_name(id_to_name_filename)
        return placement, routing, id_to_name

    import copy
    placement_save = copy.deepcopy(placement)
    routing_save = copy.deepcopy(routing)
    id_to_name_save = copy.deepcopy(id_to_name)

    routing_result_graph = construct_graph(placement, routing, id_to_name, netlist)

    verify_graph(graph)
    max_itr = None 
    curr_freq = 0
    itr = 0
    curr_freq, crit_path, crit_nets = sta(graph)
    while max_itr == None:
            try:
                kernel_latencies = update_kernel_latencies(app_dir, graph, id_to_name, placement, routing)
                break_crit_path(graph, id_to_name, crit_path, placement, routing)
                curr_freq, crit_path, crit_nets = sta(graph)
                graph.regs = None
                kernel_latencies = update_kernel_latencies(app_dir, graph, id_to_name, placement, routing)
            except:
                max_itr = itr
            itr += 1

    id_to_name = id_to_name_save
    placement = placement_save
    routing = routing_save
    graph = construct_graph(placement, routing, id_to_name)
    verify_graph(graph)
    curr_freq, crit_path, crit_nets = sta(graph)

    for _ in range(max_itr):
            break_crit_path(graph, id_to_name, crit_path, placement, routing)
            curr_freq, crit_path, crit_nets = sta(graph)

    graph.regs = None

    kernel_latencies = update_kernel_latencies(app_dir, graph, id_to_name, placement, routing)

    freq_file = os.path.join(app_dir, "design.freq")
    fout = open(freq_file, "w")
    fout.write(f"{curr_freq}\n")

    dump_routing_result(app_dir, routing) 
    dump_placement_result(app_dir, placement, id_to_name)
    dump_id_to_name(app_dir, id_to_name)

    visualize_pnr(graph, crit_nets)

    return placement, routing, id_to_name
