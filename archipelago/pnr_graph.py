import sys
import os
import argparse
import re
import itertools
import glob
import json
 
from graphviz import Digraph

from pycyclone.io import load_placement
from canal.pnr_io import __parse_raw_routing_result
from typing import Dict, List, NoReturn, Tuple, Set


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
        for in_node in self.inputs:
            queue = []
            visited = set()
            kernel = self.get_node(in_node).kernel
            queue.append(in_node)
            visited.add(in_node)
            while queue:
                n = queue.pop()
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

