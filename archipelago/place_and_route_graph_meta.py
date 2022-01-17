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
from .new_visualizer import visualize_pnr


class Node:

    def __init__(self, type_, x, y, tile_id = None, route_type = None, track = None, side = None, io = None, bit_width = None, port = None, net_id = None, reg_name = None, rmux_name = None, reg = False, kernel = None):
        assert x != None
        assert y != None
        if type_ == "tile":
            assert tile_id != None
            self.tile_id = tile_id 
        elif type_ == "route":
            assert bit_width != None
            self.tile_id = f"{type_ or 0},{route_type or 0},{x or 0},{y or 0},{track or 0},{side or 0},{io or 0},{bit_width or 0},{port or 0},{net_id or 0},{reg_name or 0},{rmux_name or 0},{reg}"
        assert self.tile_id != None
        self.type_ = type_
        self.route_type = route_type
        self.track = track 
        self.x = x 
        self.y = y 
        self.side = side 
        self.io = io 
        self.bit_width = bit_width 
        self.port = port 
        self.net_id = net_id
        self.reg_name = reg_name
        self.rmux_name = rmux_name
        self.reg = reg
        self.kernel = kernel
    
    def update_tile_id(self):
        if self.type_ == "tile":
            assert self.tile_id != None
            self.tile_id = self.tile_id 
        elif self.type_ == "route":
            assert self.bit_width != None
            self.tile_id = f"{self.type_ or 0},{self.route_type or 0},{self.x or 0},{self.y or 0},{self.track or 0},{self.side or 0},{self.io or 0},{self.bit_width or 0},{self.port or 0},{self.net_id or 0},{self.reg_name or 0},{self.rmux_name or 0},{self.reg}"
        assert self.tile_id != None
    def to_route(self):
        assert self.type_ == 'route'

        if self.route_type == "SB":
            route_string = f"{self.route_type} ({self.track}, {self.x}, {self.y}, {self.side}, {self.io}, {self.bit_width})"
        elif self.route_type == "PORT":
            route_string = f"{self.route_type} ({self.port}, {self.x}, {self.y}, {self.bit_width})"
        elif self.route_type == "REG":
            route_string = f"{self.route_type} ({self.reg_name}, {self.track}, {self.x}, {self.y}, {self.bit_width})"
        elif self.route_type == "RMUX":
            route_string = f"{self.route_type} ({self.rmux_name}, {self.x}, {self.y}, {self.bit_width})"
        else:
            raise ValueError("Unrecognized route type")
        return route_string

    def to_route(self):
        assert self.type_ == 'route'

        if self.route_type == "SB":
            route = [self.route_type, self.track, self.x, self.y, self.side, self.io, self.bit_width]
        elif self.route_type == "PORT":
            route = [self.route_type, self.port, self.x, self.y, self.bit_width]
        elif self.route_type == "REG":
            route = [self.route_type, self.reg_name, self.track, self.x, self.y, self.bit_width]
        elif self.route_type == "RMUX":
            route = [self.route_type, self.rmux_name, self.x, self.y, self.bit_width]
        else:
            raise ValueError("Unrecognized route type")
        return route

    def to_string(self):
        if self.type_ == "tile":
            return f"{self.tile_id} x:{self.x} y:{self.y} {self.kernel}"
        else:
            return f"{self.route_type} x:{self.x} y:{self.y}\nt:{self.track} bw:{self.bit_width} n:{self.net_id}\np:{self.port} r:{self.reg} {self.kernel}"

class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[(str, str)] = []
        self.edge_weights: Dict[(str, str), int] = {}
        self.inputs:List[str] = []
        self.outputs: List[str] = []
        self.sources: Dict[str, List[str]] = {}
        self.sinks: Dict[str, List[str]] = {}
        self.id_to_name: Dict[str, str] = {}
        self.node_latencies: Dict[str, int] = {}
        self.added_regs = 0
        self.mems = None
        self.pes = None
        self.ponds = None
        self.input_ios = None
        self.output_ios = None
        self.regs = None
        self.shift_regs = None
        self.roms = None
    
    def get_node(self, node: str):
        if node in self.nodes:
            return self.nodes[node]
        return None

    def get_tiles(self):
        tiles = []
        for node in self.nodes:
            if self.get_node(node).type_ == "tile":
                tiles.append(node)
        return tiles

    def get_mems(self):
        if not self.mems:
            mems = []
            for node in self.nodes:
                if self.get_node(node).type_ == "tile" and self.get_node(node).tile_id[0] == 'm':
                    mems.append(node)
            self.mems = mems
        return self.mems

    def get_roms(self):
        if not self.roms:
            mems = []
            for node in self.nodes:
                if self.get_node(node).type_ == "tile" and self.get_node(node).tile_id[0] == 'm':
                    rom = False
                    for source in self.sources[node]:
                        if self.get_node(source).port == "ren_in_0":
                            rom = True
                            break
                    if rom:
                        mems.append(node)
            self.roms = mems
        return self.roms


    def get_regs(self):
        if not self.regs:
            regs = []
            for node in self.nodes:
                if self.get_node(node).type_ == "tile" and self.get_node(node).tile_id[0] == 'r':
                    regs.append(node)
            self.regs = regs
        return self.regs

    def get_shift_regs(self):
        if not self.shift_regs:
            regs = []
            for node in self.nodes:
                if self.get_node(node).type_ == "tile" and self.get_node(node).tile_id[0] == 'r' and "d_reg_" in self.id_to_name[node]:
                    regs.append(node)
            self.shift_regs = regs
        return self.shift_regs

    def get_ponds(self):
        if not self.ponds:
            ponds = []
            for node in self.nodes:
                if self.get_node(node).type_ == "tile" and self.get_node(node).tile_id[0] == 'M':
                    ponds.append(node)
            self.ponds = ponds
        return self.ponds

    def get_pes(self):
        if not self.pes:
            pes = []
            for node in self.nodes:
                if self.get_node(node).type_ == "tile" and self.get_node(node).tile_id[0] == 'p':
                    pes.append(node)
            self.pes = pes
        return self.pes

    def get_input_ios(self):
        if not self.input_ios:
            ios = []
            for node in self.nodes:
                if self.get_node(node).type_ == "tile" and (self.get_node(node).tile_id[0] == 'I' or self.get_node(node).tile_id[0] == 'i') and len(self.sources[node]) == 0:
                    ios.append(node)
            self.input_ios = ios
        return self.input_ios

    def get_output_ios(self):
        if not self.output_ios:
            ios = []
            for node in self.nodes:
                if self.get_node(node).type_ == "tile" and (self.get_node(node).tile_id[0] == 'I' or self.get_node(node).tile_id[0] == 'i') and len(self.sinks[node]) == 0:
                    ios.append(node)
            self.output_ios = ios
        return self.output_ios

    def is_reachable(self, source, dest):
        visited = set()
        queue = []

        queue.append(source)
        visited.add(source)

        while queue:
            n = queue.pop()

            if n == dest:
                return True

            for node in self.sinks[n]:
                if node not in visited:
                    queue.append(node)
                    visited.add(node)
        return False


    def add_node(self, node: Node):
        self.nodes[node.tile_id] = node

    def add_edge(self, node1, node2):
        if type(node1) == str:
            node1_name = node1
        elif type(node1) == Node:
            node1_name = node1.tile_id
        else:
            raise TypeError(f"Source node is type {type(node1)}")

        if type(node2) == str:
            node2_name = node2
        elif type(node2) == Node:
            node2_name = node2.tile_id
        else:
            raise TypeError(f"Dest node is type {type(node2)}")

        assert node1_name in self.nodes, f"{node1_name} not in nodes"
        assert node2_name in self.nodes, f"{node2_name} not in nodes"
        if (node1_name, node2_name) not in self.edges:
            self.edges.append((node1_name, node2_name))

    def update_sources_and_sinks(self):
        new_nodes = {}
        for node in self.nodes:
            g_node = self.get_node(node)
            g_node.update_tile_id()
            new_nodes[g_node.tile_id] = g_node

        new_edges = []
        for (node1_name, node2_name) in self.edges:
            g_node_1 = self.get_node(node1_name)
            g_node_2 = self.get_node(node2_name)
            g_node_1.update_tile_id()
            g_node_2.update_tile_id()
            
            new_edges.append((g_node_1.tile_id, g_node_2.tile_id))

        self.nodes = new_nodes
        self.edges = new_edges

        self.inputs = []
        self.outputs = []
        for node in self.nodes:
            self.sources[node] = []
            self.sinks[node] = []
        for node in self.nodes:
            for source,sink in self.edges:
                if node == source:
                    self.sources[sink].append(source)
                elif node == sink:
                    self.sinks[source].append(sink)
        for node in self.nodes:
            if len(self.sources[node]) == 0:
                self.inputs.append(node)
            if len(self.sinks[node]) == 0:
                self.outputs.append(node)

    def update_edge_kernels(self):

        # Fix input kernel names  
        # for out_node in self.get_mems() + self.get_pes():
        #     queue = []
        #     visited = set()
        #     kernel = self.get_node(out_node).kernel
        #     queue.append(out_node)
        #     visited.add(out_node)
        #     while queue:
        #         n = queue.pop()
        #         kernel = self.get_node(n).kernel

        #         for node in self.sources[n]:
        #             if node not in visited:
        #                 queue.append(node)
        #                 visited.add(node)
        #                 if self.get_node(node).type_ == "route" and not self.get_node(node).kernel == 'reset':
        #                     self.get_node(node).kernel = kernel 


        for in_node in self.inputs:
            queue = []
            visited = set()
            kernel = self.get_node(in_node).kernel
            queue.append(in_node)
            visited.add(in_node)
            while queue:
                n = queue.pop()
                # if self.get_node(n).type_ == "tile":
                kernel = self.get_node(n).kernel

                for node in self.sinks[n]:
                    if node not in visited:
                        queue.append(node)
                        visited.add(node)
                        if self.get_node(node).type_ == "route":
                            self.get_node(node).kernel = kernel


        for tile in self.get_tiles():
            for source in self.sources[tile]:
                self.get_node(source).kernel = self.get_node(tile).kernel


    def print_graph(self, filename, edge_weights = False):
        g = Digraph()
        for node in self.nodes:
            g.node(node, label = self.get_node(node).to_string())

        for edge in self.edges:
            if self.get_node(edge[0]).net_id != None:
                net_id = self.get_node(edge[0]).net_id
            else:
                net_id = self.get_node(edge[1]).net_id

            if edge_weights:
                g.edge(edge[0], edge[1], label=str(self.edge_weights[edge]))
            else:
                g.edge(edge[0], edge[1], label=net_id)
            
        g.render(filename=filename)

    def print_graph_tiles_only(self, filename):
        g = Digraph()
        for source in self.get_tiles():
            if source[0] == 'r':
                g.node(source, label = f"{source}\n{self.get_node(source).kernel}", shape='box')
            else:
                g.node(source, label = f"{source}\n{self.get_node(source).kernel}")
            for dest in self.get_tiles():
                reachable = False
                visited = set()
                queue = []
                queue.append(source)
                visited.add(source)
                while queue:
                    n = queue.pop()

                    if n == dest and n != source:
                        reachable = True

                    for node in self.sinks[n]:
                        if node not in visited:
                            if self.get_node(node).type_ == "tile":
                                if node == dest:
                                    reachable = True
                            else:
                                queue.append(node)
                                visited.add(node)

                if reachable:
                    if self.get_node(source).net_id != None:
                        net_id = self.get_node(source).net_id
                    else:
                        net_id = self.get_node(dest).net_id
                    g.edge(source, dest, label=net_id)
            
        g.render(filename=filename)


    def topological_sort(self):
        visited = set()
        stack: List[str] = []
        for n in self.inputs:
            if n not in visited:
                self.topological_sort_helper(n, stack, visited)
        return stack[::-1]

    def topological_sort_helper(self, node: str, stack, visited: Set[str]):
        visited.add(node)
        for ns in self.sinks[node]:
            if ns not in visited:
                self.topological_sort_helper(ns, stack, visited)
        stack.append(node)

    def removeEdge(self, edge):
        node0 = edge[0]
        node1 = edge[1]

        if edge in self.edges:
            self.edges.remove(edge)
        if node0 in self.sources[node1]:
            self.sources[node1].remove(node0)
        if node1 in self.sinks[node0]:
            self.sinks[node0].remove(node1)


    def isCyclicUtil(self, v, visited, recStack):
        visited.append(v)
        recStack.append(v)
  
        for neighbour in self.sinks[v]:
            if neighbour not in visited:
                retval = self.isCyclicUtil(neighbour, visited, recStack)
                if retval != None:
                    return retval
            elif neighbour in recStack:
                return (v, neighbour)

        recStack.remove(v)
        return None
  
    def FixCycles(self):
        sys.setrecursionlimit(10**5)
        visited = []
        recStack = []
        for node in self.inputs:
            if node not in visited:
                break_edge = self.isCyclicUtil(node,visited,recStack)
                if break_edge != None:
                    print("Breaking cycle at", break_edge[1])
                    self.removeEdge(break_edge)
                    return True
        return False

    def get_outputs_of_kernel(self, kernel):
        kernel_nodes = set()
        for node in self.nodes:
            if self.get_node(node).kernel == kernel:
                kernel_nodes.add(node)

        kernel_output_nodes = set()

        for source in kernel_nodes:
            visited = set()
            queue = []

            queue.append(source)
            visited.add(source)

            while queue:
                n = queue.pop()

                if self.get_node(n).kernel != kernel:
                    kernel_output_nodes.add(source)
                    break
                elif n != source:
                    continue

                for node in self.sinks[n]:
                    if node not in visited:
                        queue.append(node)
                        visited.add(node)
        return kernel_output_nodes

    def get_output_tiles_of_kernel(self, kernel):
        if kernel in self.nodes:
            kernel = self.get_node(kernel).kernel
        kernel_nodes = set()
        for node in self.nodes:
            if self.get_node(node).type_ == "tile" and self.get_node(node).kernel == kernel:
                kernel_nodes.add(node)

        kernel_output_nodes = set()

        for source in kernel_nodes:
            visited = set()
            queue = []

            queue.append(source)
            visited.add(source)

            while queue:
                n = queue.pop()

                if self.get_node(n).type_ == "tile" and self.get_node(n).kernel != kernel:
                    kernel_output_nodes.add(source)
                    break
                elif n != source and self.get_node(n).type_ == "tile":
                    continue

                for node in self.sinks[n]:
                    if node not in visited:
                        queue.append(node)
                        visited.add(node)
        return kernel_output_nodes


class KernelNode:
    def __init__(self, mem_id = None, kernel = None, type_ = None, latency = 0, flush_latency = 0, has_shift_regs = False):
        self.mem_id = mem_id
        self.kernel = kernel
        self.type_ = type_
        self.latency = latency
        self.flush_latency = flush_latency
        self.has_shift_regs = has_shift_regs

    def to_string(self):
        if self.kernel:
            return f"{self.kernel} {self.type_} {self.latency} {self.flush_latency}"
        else:
            return f"{self.mem_id} {self.type_} {self.latency} {self.flush_latency}"

class KernelGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[(str, str)] = []
        self.inputs:List[str] = []
        self.outputs: List[str] = []
        self.sources: Dict[str, List[str]] = {}
        self.sinks: Dict[str, List[str]] = {}

    def get_node(self, node: str):
        if node in self.nodes:
            return self.nodes[node]
        return None

    def is_reachable(self, source, dest):
        visited = set()
        queue = []

        queue.append(source)
        visited.add(source)

        while queue:
            n = queue.pop()

            if n == dest:
                return True

            for node in self.sinks[n]:
                if node not in visited:
                    queue.append(node)
                    visited.add(node)
        return False


    def add_node(self, node: KernelNode):
        if node.kernel:
            self.nodes[node.kernel] = node
        else:
            self.nodes[node.mem_id] = node

    def add_edge(self, node1, node2):
        if type(node1) == str:
            node1_name = node1
        elif type(node1) == KernelNode:
            if node1.kernel:
                node1_name = node1.kernel
            else:
                node1_name = node1.mem_id
        else:
            raise TypeError(f"Source node is type {type(node1)}")

        if type(node2) == str:
            node2_name = node2
        elif type(node2) == KernelNode:
            if node2.kernel:
                node2_name = node2.kernel
            else:
                node2_name = node2.mem_id
        else:
            raise TypeError(f"Dest node is type {type(node2)}")

        assert node1_name in self.nodes, f"{node1_name} not in nodes"
        assert node2_name in self.nodes, f"{node2_name} not in nodes"
        if (node1_name, node2_name) not in self.edges:
            self.edges.append((node1_name, node2_name))

    def update_sources_and_sinks(self):
        self.inputs = []
        self.outputs = []
        for node in self.nodes:
            self.sources[node] = []
            self.sinks[node] = []
        for node in self.nodes:
            for source,sink in self.edges:
                if node == source:
                    self.sources[sink].append(source)
                elif node == sink:
                    self.sinks[source].append(sink)
        for node in self.nodes:
            if len(self.sources[node]) == 0:
                self.inputs.append(node)
            if len(self.sinks[node]) == 0:
                self.outputs.append(node)

    def topological_sort(self):
        visited = set()
        stack: List[str] = []
        for n in self.inputs:
            if n not in visited:
                self.topological_sort_helper(n, stack, visited)
        return stack[::-1]

    def topological_sort_helper(self, node: str, stack, visited: Set[str]):
        visited.add(node)
        for ns in self.sinks[node]:
            if ns not in visited:
                self.topological_sort_helper(ns, stack, visited)
        stack.append(node)

    def print_graph(self, filename):
        g = Digraph()
        for node in self.nodes:
            g.node(node, label = self.get_node(node).to_string())

        for edge in self.edges:
            g.edge(edge[0], edge[1])
            
        g.render(filename=filename)

def parse_args():
    parser = argparse.ArgumentParser("CGRA Retiming tool")
    parser.add_argument("-a", "--app", "-d", required=True, dest="application", type=str, help="Application directory")
    parser.add_argument("-f", "--min-frequency", default=200, dest="frequency", type=int,
                        help="Minimum frequency in MHz")
    args = parser.parse_args()
    # check filenames
    # assert 1000 > args.frequency > 0, "Frequency must be less than 1GHz"
    dirname = os.path.join(args.application, "bin")
    netlist = os.path.join(dirname, "design.packed")
    assert os.path.exists(netlist), netlist + " does not exist"
    placement = os.path.join(dirname, "design.place")
    assert os.path.exists(placement), placement + " does not exists"
    route = os.path.join(dirname, "design.route")
    assert os.path.exists(route), route + " does not exists"
    # need to load routing files as well
    # for now we just assume RMUX exists
    return netlist, placement, route, args.frequency

def load_netlist(netlist_file):
    f = open(netlist_file, "r")
    lines = f.readlines()
 
    netlist = {}
    id_to_name = {}
    netlist_read = False
    id_to_name_read = False

    for line in lines:
        if "Netlists:" in line:
            netlist_read = True
        elif "ID to Names:" in line:
            netlist_read = False
            id_to_name_read = True
        elif "Netlist Bus:" in line:
            netlist_read = False
            id_to_name_read = False
        elif netlist_read:
            if len(line.split(":")) > 1:
                edge_id = line.split(":")[0]
                connections = line.split(":")[1]

                connections = re.findall(r'\b\S+\b', connections)

                netlist[edge_id] = []
                for conn1, conn2 in zip(connections[::2], connections[1::2]):
                    netlist[edge_id].append((conn1, conn2))
        elif id_to_name_read:
            if len(line.split(":")) > 1:
                id = line.split(":")[0]
                name = line.split(":")[1]
                id_to_name[id] = name.strip()
              
    return netlist, id_to_name

def load_folded_regs(folded_file):
    f = open(folded_file, "r")
    lines = f.readlines()
    pe_reg = set()
 
    for line in lines:
        reg_entry = re.findall(r'\b\S+\b', line.split(":")[0])
        entry = re.findall(r'\b\S+\b', line.split(":")[1])
        blk_id = entry[0]
        port = entry[-1]
        if reg_entry[0][0] == 'r' and blk_id[0] == 'p':
            pe_reg.add(((reg_entry[0], reg_entry[1]),(blk_id, port)))

    return pe_reg

def load_shift_regs(shift_regs_file, pe_reg):
    shift_regs = set()
    folded_regs = {reg:pe for (reg,_),pe in pe_reg}

    f = open(shift_regs_file, "r")
    lines = f.readlines()
    pe_reg = set()
 
    for line in lines:
        id = line.strip()
        shift_regs.add((id, None))
        if id in folded_regs:
            shift_regs.add(folded_regs[id])        
    return shift_regs

def segment_to_node(segment, net_id):
    if segment[0] == "SB":
        track, x, y, side, io_, bit_width = segment[1:]
        node1 = Node("route", x, y, route_type = "SB", track = track, side = side, io = io_, bit_width = bit_width, net_id = net_id)
    elif segment[0] == "PORT":
        port_name, x, y, bit_width = segment[1:]
        node1 = Node("route", x, y, route_type = "PORT", bit_width = bit_width, net_id = net_id, port = port_name)
    elif segment[0] == "REG":
        reg_name, track, x, y, bit_width = segment[1:]
        node1 = Node("route", x, y, route_type = "REG", track = track, bit_width = bit_width, net_id = net_id, reg_name = reg_name)
    elif segment[0] == "RMUX":
        rmux_name, x, y, bit_width = segment[1:]
        node1 = Node("route", x, y, route_type = "RMUX", bit_width = bit_width, net_id = net_id, rmux_name = rmux_name)
    else:
        raise ValueError("Unrecognized route type")
    return node1

def get_tile_at(x, y, bw, placement, port = "", reg = False):

    pond_ports = ["data_in_pond", "data_out_pond", "flush"]

    for tile_id, place in placement.items():
        if (x,y) == place:
            if reg:
                if tile_id[0] == 'r':
                    return tile_id
            elif y == 0:
                assert tile_id[0] == "I" or tile_id[0] == "i"
                if tile_id[0] == "I" and bw == 16:
                    return tile_id
                elif tile_id[0] == "i" and bw == 1:
                    return tile_id
            elif (x+1) % 4 == 0:
                assert tile_id[0] == "m"
                return tile_id
            else:
                assert tile_id[0] == "M" or tile_id[0] == "p", tile_id
                if port in pond_ports:
                    if tile_id[0] == "M":
                        return tile_id
                else:         
                    if tile_id[0] != "M":             
                        return tile_id      

    return None


def construct_graph(placement, routes, id_to_name):
    graph = Graph()
    graph.id_to_name = id_to_name
    max_reg_id = 0
    

    for blk_id, place in placement.items():
        if len(graph.id_to_name[blk_id].split("$")) > 0:
            kernel = graph.id_to_name[blk_id].split("$")[0]
        else:
            kernel = None
        node = Node("tile", place[0], place[1], tile_id=blk_id, kernel = kernel)
        graph.add_node(node)
        max_reg_id = max(max_reg_id, int(blk_id[1:]))
    graph.added_regs = max_reg_id + 1

    for net_id, net in routes.items():
        if net_id == "e0" and 'POND_PIPELINED' in os.environ and os.environ['POND_PIPELINED'] == '1':
            continue
        for route in net:
            for seg1, seg2 in zip(route, route[1:]):
                node1 = segment_to_node(seg1, net_id)
                graph.add_node(node1)
                node2 = segment_to_node(seg2, net_id)
                graph.add_node(node2)
                graph.add_edge(node1, node2)
                
                if node1.route_type == "PORT":
                    tile_id = get_tile_at(node1.x, node1.y, node1.bit_width, placement, node1.port)
                    if (tile_id[0] == "m" and node1.port != "flush"):
                        node2.reg = True
                    graph.add_edge(tile_id, node1)
                elif node1.route_type == "REG":
                    tile_id = get_tile_at(node1.x, node1.y, node1.bit_width, placement, reg = True)
                    graph.add_edge(tile_id, node1)

                if node2.route_type == "PORT":
                    tile_id = get_tile_at(node2.x, node2.y, node2.bit_width, placement, node2.port)
                    #if tile_id[0] == "m":
                    #     node2.reg = True
                    graph.add_edge(node2, tile_id)
                elif node2.route_type == "REG":
                    tile_id = get_tile_at(node2.x, node2.y, node2.bit_width, placement, reg = True)
                    node2.reg = True
                    graph.add_edge(node2, tile_id)

    
    graph.update_sources_and_sinks()
    graph.update_edge_kernels()
    if 'POND_PIPELINED' in os.environ and os.environ['POND_PIPELINED'] == '1':
        for pe in graph.get_ponds()+graph.get_pes():
            sources = graph.sources[pe]
            for source in sources:
                graph.get_node(source).reg = True
    elif 'PIPELINED' in os.environ and os.environ['PIPELINED'] == '1':
        for pe in graph.get_pes():
            sources = graph.sources[pe]
            for source in sources:
                graph.get_node(source).reg = True


    graph.update_sources_and_sinks()
    graph.update_edge_kernels()

    for node in graph.nodes:
        if graph.get_node(node).reg:
            graph.node_latencies[node] = 1
        else:
            graph.node_latencies[node] = 0

    return graph

def construct_tile_graph(graph):
    tile_graph = Graph()
    
    for source in graph.get_tiles():
        tile_graph.add_node(graph.get_node(source))

        for dest in graph.get_tiles():
            num_avail_regs = 0
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
                    if graph.get_node(n).route_type == "RMUX" and graph.get_node(node).route_type == "SB":
                        num_avail_regs += 1

                    if node not in visited:
                        if graph.get_node(node).type_ == "tile":
                            if node == dest:
                                reachable = True
                        else:
                            queue.append(node)
                            visited.add(node)
            if reachable:
                if graph.get_node(source).net_id != None:
                    net_id = graph.get_node(source).net_id
                else:
                    net_id = graph.get_node(dest).net_id
                tile_graph.add_node(graph.get_node(dest))
                tile_graph.add_edge(source, dest)

                tile_graph.edge_weights[(source, dest)] = num_avail_regs
    
    return tile_graph

def verify_graph(graph):
    #for node in graph.nodes:
    #    if len(graph.sources[node]) == 0:
    #        assert node in graph.inputs
    #        assert graph.get_node(node).type_ == "tile", f"{node} is a route"
    #        assert graph.get_node(node).tile_id[0] == "I" or graph.get_node(node).tile_id[0] == "i" or graph.get_node(node).tile_id[0] == "p"
    #    if len(graph.sinks[node]) == 0:
    #        assert node in graph.outputs
    #        assert graph.get_node(node).type_ == "tile", f"{node} is a route"
    #        assert graph.get_node(node).tile_id[0] == "I" or graph.get_node(node).tile_id[0] == "i"

    while graph.FixCycles():
        pass
    
    
PE_DELAY = 700
MEM_DELAY = 0
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
        # print(curr_node, timing_info[curr_node].get_total(), "glb:", timing_info[curr_node].glbs, "hops:",  timing_info[curr_node].hops, "pes:", timing_info[curr_node].pes, "mems:", timing_info[curr_node].mems, timing_info[curr_node].available_regs)

    crit_path.reverse()

    print("\tCritical Path Nets:", *net_ids)
    
    # for idx, curr_node in enumerate(node_to_timing):
    #     if graph.get_node(curr_node).type_ == "route":
    #         continue
    #     if idx > 500:
    #         break
    #     print(curr_node, timing_info[curr_node].get_total(), timing_info[curr_node].glbs, timing_info[curr_node].hops, timing_info[curr_node].pes, timing_info[curr_node].mems, timing_info[curr_node].available_regs)
    return clock_speed, crit_path, crit_edges



def find_break_idx(graph, crit_path):
    # print(crit_path)
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
   
    # print(f"Breaking between {g_node.kernel} and {g_sink.kernel}")
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
                # if g_node.reg:
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
            print(f"Incorrect node delay: {node} {cycles}")
            # print(f"Fixing branching delays at: {node} {cycles}")
            # source_cycles = [node_cycles[source] for source in graph.sources[node]]
            # min_parent_cycles = min(source_cycles)
            # for parent in graph.sources[node]:
            #     if node_cycles[parent] == min_parent_cycles:
            #         break_at(graph, graph.sources[parent][0], id_to_name, placement, routing)
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
                      
            if parent in graph.get_mems():
                cycles.add(0)
            elif not (g_parent.type_ == "route" and g_parent.port == "flush"):
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
                if node_cycles[g_node.kernel][parent] != max_parent_cycles:
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

            #if "reset" not in parent and not (g_parent.type_ == "mem" and not g_parent.has_shift_regs):
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
 
    outputs = [output for output in kernel_graph.outputs if output[0] == "I"]
    bitoutputs = [output for output in kernel_graph.outputs if output[0] == "i"]
    output_nodes = [(kernel_graph.sources[output][0], graph.sources[output][0]) for output in outputs]
    bitoutputs_nodes = [(kernel_graph.sources[output][0], graph.sources[output][0]) for output in bitoutputs]

    kernel_output_delays = set()
    kernel_output_nodes_and_delays = []
    for output_node, graph_output_node in output_nodes:
        if node_cycles[output_node]: 
            kernel_output_delays.add(node_cycles[output_node])
            kernel_output_nodes_and_delays.append((graph_output_node, node_cycles[output_node]))

    if len(kernel_output_delays) > 1:
        print(f"\tIncorrect delay at output of application: {kernel_output_delays}")
        # breakpoint()
        max_parent_cycles = max(kernel_output_delays)
        for node, delay in kernel_output_nodes_and_delays:
            if delay != max_parent_cycles:
                print(f"\tFixing branching delays at: {node} {max_parent_cycles - delay}") 
                for _ in range(max_parent_cycles - delay):
                    break_at(graph, node, id_to_name, placement, routing)

    kernel_output_delays = set()
    kernel_output_nodes_and_delays = []
    for output_node, graph_output_node in bitoutputs_nodes:
        if node_cycles[output_node]: 
            kernel_output_delays.add(node_cycles[output_node])
            kernel_output_nodes_and_delays.append((graph_output_node, node_cycles[output_node]))

    if len(kernel_output_delays) > 1:
        print(f"\tIncorrect delay at output of application: {kernel_output_delays}")
        max_parent_cycles = max(kernel_output_delays)
        for node, delay in kernel_output_nodes_and_delays:
            if delay != max_parent_cycles:
                print(f"\tFixing branching delays at: {node} {max_parent_cycles - delay}") 
                for _ in range(max_parent_cycles - delay):
                    break_at(graph, node, id_to_name, placement, routing)


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

    # print("Kernel cycles:")

    kernel_latencies = {}

    for kernel in node_cycles:
        kernel_cycles = [cyc for cyc in node_cycles[kernel].values() if cyc != None]
        kernel_cycles.append(0)
        # print("\t", kernel,  max(kernel_cycles))
        kernel_latencies[kernel] = max(kernel_cycles)

    # print("\n")
    return kernel_latencies


def flush_cycles(graph):
    nodes = graph.topological_sort()
    node_cycles = {}
    # kernel_latencies = {}

    for io in graph.get_input_ios():
        if graph.get_node(io).kernel == "io1in_reset":
            break
    assert graph.get_node(io).kernel == "io1in_reset"
    flush_cycles = {}
    # print("Flush cycles:")

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

            #assert len(graph.sources[curr_node]) == 1, f"{mem} {graph.sources[curr_node]}"
            curr_node = graph.sources[curr_node][0]

        # kernel_latencies[graph.get_node(mem).kernel] = flush_cycles[mem]

    # print("\n")
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


    # kernel_graph = KernelGraph()

    # for source, dest in graph.edges:
    #     source_kernel = graph.get_node(source).kernel
    #     if source_kernel not in kernel_graph.nodes:
    #         kernel_graph.add_node(KernelNode(kernel = source_kernel))
    #     if source in graph.get_pes():
    #         kernel_graph.get_node(source_kernel).type_ = "compute"
    #     elif source in graph.get_mems():
    #         kernel_graph.get_node(source_kernel).type_ = "mem"
    #     elif source in graph.get_regs():
    #         kernel_graph.get_node(source_kernel).has_shift_regs = True
    #     if source_kernel in new_latencies:
    #         kernel_graph.get_node(source_kernel).latency = new_latencies[source_kernel]
    #     if source_kernel in flush_latencies:
    #         kernel_graph.get_node(source_kernel).flush_latency = flush_latencies[source_kernel]


    #     dest_kernel = graph.get_node(dest).kernel
    #     if dest_kernel not in kernel_graph.nodes:
    #         kernel_graph.add_node(KernelNode(kernel = dest_kernel))
    #     if dest in graph.get_pes():
    #         kernel_graph.get_node(dest_kernel).type_ = "compute"
    #     elif dest in graph.get_mems():
    #         kernel_graph.get_node(dest_kernel).type_ = "mem"
    #     elif dest in graph.get_regs():
    #         kernel_graph.get_node(dest_kernel).has_shift_regs = True
    #     if dest_kernel in new_latencies:
    #         kernel_graph.get_node(dest_kernel).latency = new_latencies[dest_kernel]
    #     if dest_kernel in flush_latencies:
    #         kernel_graph.get_node(dest_kernel).flush_latency = flush_latencies[dest_kernel]


    #     if (source_kernel, dest_kernel) not in kernel_graph.edges and source_kernel != dest_kernel:
    #         kernel_graph.add_edge(source_kernel, dest_kernel)

    # kernel_graph.update_sources_and_sinks()
    # kernel_graph.print_graph("kernel_graph")
    # return kernel_graph



def calculate_latencies(kernel_graph, kernel_latencies):

    nodes = kernel_graph.topological_sort()

#    for f_node in nodes:
#        flush_latency = kernel_graph.get_node(f_node).flush_latency
        
#        if "hw_input" in f_node and "global_wrapper_stencil" in f_node and flush_latency != 0:
#            kernel_graph.get_node(f_node).flush_latency = 0
#            for node in kernel_graph.nodes:
#                if node != f_node and kernel_graph.is_reachable(f_node, node):
#                    if kernel_graph.get_node(node).type_ == "mem":
#                        kernel_graph.get_node(node).flush_latency -= flush_latency

    kernel_graph.print_graph("kernel_graph_updated")

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
        new_lat = lat

        if f"op_{kernel}" in sorted_new_latencies:
            new_lat = sorted_new_latencies[f"op_{kernel}"]
            # print("Found exact kernel:", kernel)
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
                    # print("Did not find kernel", kernel)
                    new_lat = None
            # print("Found inexact kernel:", kernel, f_kernel)
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
    #branch_delay_match_kernels(kernel_graph, graph, id_to_name, placement, routing)

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
    


#    for kernel, lat in kernel_latencies.items():
#        if "global_wrapper_stencil" in kernel and "hw_input" in kernel:
#            f_kernel_name = kernel.split("hcompute_")[1]
#            if f_kernel_name in flush_latencies:
#                kernel_latencies[kernel] -= flush_latencies[f_kernel_name]
#                flush_latencies[f_kernel_name] = 0
                #breakpoint()

    # print("Final Kernel Latencies")
    # for kernel, latency in kernel_latencies.items():
    #     print("\t", kernel, latency)

    # print("Final Flush Latencies")
    # for kernel, latency in flush_latencies.items():
    #     print("\t", kernel, latency)

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
     

def pipeline_pnr(app_dir, placement, routing, id_to_name, target_freq, load_only):
    if load_only:
        id_to_name_filename = os.path.join(app_dir, f"design.id_to_name")
        if os.path.isfile(id_to_name_filename):
            id_to_name = load_id_to_name(id_to_name_filename)
        return placement, routing, id_to_name

    import copy
    placement_save = copy.deepcopy(placement)
    routing_save = copy.deepcopy(routing)
    id_to_name_save = copy.deepcopy(id_to_name)

    graph = construct_graph(placement, routing, id_to_name)
    graph.print_graph_tiles_only("pnr_graph_tile")
    #graph.print_graph("pnr_graph")
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
                print("max_itr set")
                max_itr = itr
            itr += 1

    id_to_name = id_to_name_save
    placement = placement_save
    routing = routing_save
    graph = construct_graph(placement, routing, id_to_name)
    graph.print_graph_tiles_only("pnr_graph_tile")
    #graph.print_graph("pnr_graph")
    verify_graph(graph)
    curr_freq, crit_path, crit_nets = sta(graph)

    for _ in range(max_itr):
            break_crit_path(graph, id_to_name, crit_path, placement, routing)
            curr_freq, crit_path, crit_nets = sta(graph)

    graph.regs = None

    graph.print_graph_tiles_only("pnr_graph_tile_post_pipe")
            
    kernel_latencies = update_kernel_latencies(app_dir, graph, id_to_name, placement, routing)

    #if min(kernel_latencies.values()) < 0:
    #    pipeline_input_ios(graph, -(min(kernel_latencies.values())), id_to_name, placement, routing)
    #    kernel_latencies = update_kernel_latencies(app_dir, graph, id_to_name, placement, routing)
    freq_file = os.path.join(app_dir, "design.freq")
    fout = open(freq_file, "w")
    fout.write(f"{curr_freq}\n")

    dump_routing_result(app_dir, routing) 
    dump_placement_result(app_dir, placement, id_to_name)
    dump_id_to_name(app_dir, id_to_name)

    if False:
        print("Printing graph of pnr result")
        graph.print_graph("pnr_graph_post_pipe")
        graph.print_graph_tiles_only("pnr_graph_tile_post_pipe")
    visualize_pnr(graph, crit_nets)
    return placement, routing, id_to_name

def main():
    netlist_file, placement_file, routing_file, target_freq = parse_args()

    print("Loading netlist")
    netlist, id_to_name = load_netlist(netlist_file)
    print("Loading placement")
    placement = load_placement(placement_file)
    print("Loading routing")
    routing = __parse_raw_routing_result(routing_file)

    app_dir = os.path.dirname(netlist_file)

    pipeline_pnr(app_dir, placement, routing, id_to_name, target_freq, False)


    
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    # import cProfile, pstats, io
    # from pstats import SortKey
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
