import sys
import os
from typing import Dict, List, Set


class Node:
    def __init__(self, type_, x, y, tile_id=None, route_type=None, track=None,
                 side=None, io=None, bit_width=None, port=None, net_id=None,
                 reg_name=None, rmux_name=None, reg=False, kernel=None):
        assert x is not None
        assert y is not None
        if type_ == "tile":
            assert tile_id is not None
            self.tile_id = tile_id
        elif type_ == "route":
            assert bit_width is not None
            self.tile_id = f"{type_ or 0},{route_type or 0},{x or 0},{y or 0},"+\
            f"{track or 0},{side or 0},{io or 0},{bit_width or 0},{port or 0},"+\
            f"{net_id or 0},{reg_name or 0},{rmux_name or 0},{reg}"
        assert self.tile_id is not None
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
            assert self.tile_id is not None
            self.tile_id = self.tile_id
        elif self.type_ == "route":
            assert self.bit_width is not None
            self.tile_id = f"{self.type_ or 0},{self.route_type or 0},"+\
                           f"{self.x or 0},{self.y or 0},{self.track or 0},"+\
                           f"{self.side or 0},{self.io or 0},"+\
                           f"{self.bit_width or 0},{self.port or 0},"+\
                           f"{self.net_id or 0},{self.reg_name or 0},"+\
                           f"{self.rmux_name or 0},{self.reg}"
        assert self.tile_id is not None

    def to_route(self):
        assert self.type_ == 'route'

        if self.route_type == "SB":
            route_string = f"{self.route_type} ({self.track}, {self.x}, "+\
                           f"{self.y}, {self.side}, {self.io}, {self.bit_width})"
        elif self.route_type == "PORT":
            route_string = f"{self.route_type} ({self.port}, {self.x}, "+\
                           f"{self.y}, {self.bit_width})"
        elif self.route_type == "REG":
            route_string = f"{self.route_type} ({self.reg_name}, {self.track}, "+\
                           f"{self.x}, {self.y}, {self.bit_width})"
        elif self.route_type == "RMUX":
            route_string = f"{self.route_type} ({self.rmux_name}, {self.x}, "+\
                           f"{self.y}, {self.bit_width})"
        else:
            raise ValueError("Unrecognized route type")
        return route_string

    def to_route(self):
        assert self.type_ == 'route'

        if self.route_type == "SB":
            route = [self.route_type, self.track, self.x,
                     self.y, self.side, self.io, self.bit_width]
        elif self.route_type == "PORT":
            route = [self.route_type, self.port,
                     self.x, self.y, self.bit_width]
        elif self.route_type == "REG":
            route = [self.route_type, self.reg_name,
                     self.track, self.x, self.y, self.bit_width]
        elif self.route_type == "RMUX":
            route = [self.route_type, self.rmux_name,
                     self.x, self.y, self.bit_width]
        else:
            raise ValueError("Unrecognized route type")
        return route

    def to_string(self):
        if self.type_ == "tile":
            return f"{self.tile_id} x:{self.x} y:{self.y} {self.kernel}"
        else:
            return f"{self.route_type} x:{self.x} y:{self.y}\nt:{self.track} "+\
                   f"bw:{self.bit_width} n:{self.net_id}\np:{self.port} r:{self.reg} {self.kernel}"


class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[(str, str)] = []
        self.edge_weights: Dict[(str, str), int] = {}
        self.inputs: List[str] = []
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
                if self.get_node(node).type_ == "tile" and self.get_node(node).tile_id[0] == 'r' \
                   and "d_reg_" in self.id_to_name[node]:
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
                if self.get_node(node).type_ == "tile" and (self.get_node(node).tile_id[0] == 'I' \
                   or self.get_node(node).tile_id[0] == 'i') and len(self.sources[node]) == 0:
                    ios.append(node)
            self.input_ios = ios
        return self.input_ios

    def get_output_ios(self):
        if not self.output_ios:
            ios = []
            for node in self.nodes:
                if self.get_node(node).type_ == "tile" and (self.get_node(node).tile_id[0] == 'I' \
                   or self.get_node(node).tile_id[0] == 'i') and len(self.sinks[node]) == 0:
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
            for source, sink in self.edges:
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

    def remove_edge(self, edge):
        node0 = edge[0]
        node1 = edge[1]

        if edge in self.edges:
            self.edges.remove(edge)
        if node0 in self.sources[node1]:
            self.sources[node1].remove(node0)
        if node1 in self.sinks[node0]:
            self.sinks[node0].remove(node1)

    def is_cyclic_util(self, v, visited, rec_stack):
        visited.append(v)
        rec_stack.append(v)

        for neighbour in self.sinks[v]:
            if neighbour not in visited:
                retval = self.is_cyclic_util(neighbour, visited, rec_stack)
                if retval != None:
                    return retval
            elif neighbour in rec_stack:
                return (v, neighbour)

        rec_stack.remove(v)
        return None

    def fix_cycles(self):
        sys.setrecursionlimit(10**5)
        visited = []
        rec_stack = []
        for node in self.inputs:
            if node not in visited:
                break_edge = self.is_cyclic_util(node, visited, rec_stack)
                if break_edge is not None:
                    self.remove_edge(break_edge)
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
    def __init__(self, mem_id=None, kernel=None, type_=None, latency=0, flush_latency=0, has_shift_regs=False):
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
        self.inputs: List[str] = []
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
            for source, sink in self.edges:
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


def segment_to_node(segment, net_id):
    if segment[0] == "SB":
        track, x, y, side, io_, bit_width = segment[1:]
        node1 = Node("route", x, y, route_type="SB", track=track,
                     side=side, io=io_, bit_width=bit_width, net_id=net_id)
    elif segment[0] == "PORT":
        port_name, x, y, bit_width = segment[1:]
        node1 = Node("route", x, y, route_type="PORT",
                     bit_width=bit_width, net_id=net_id, port=port_name)
    elif segment[0] == "REG":
        reg_name, track, x, y, bit_width = segment[1:]
        node1 = Node("route", x, y, route_type="REG", track=track,
                     bit_width=bit_width, net_id=net_id, reg_name=reg_name)
    elif segment[0] == "RMUX":
        rmux_name, x, y, bit_width = segment[1:]
        node1 = Node("route", x, y, route_type="RMUX",
                     bit_width=bit_width, net_id=net_id, rmux_name=rmux_name)
    else:
        raise ValueError("Unrecognized route type")
    return node1


def get_tile_at(x, y, bw, placement, port="", reg=False):
    pond_ports = ["data_in_pond_0", "data_out_pond_0", "flush"]

    for tile_id, place in placement.items():
        if (x, y) == place:
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
        node = Node("tile", place[0], place[1], tile_id=blk_id, kernel=kernel)
        graph.add_node(node)
        max_reg_id = max(max_reg_id, int(blk_id[1:]))
    graph.added_regs = max_reg_id + 1

    for net_id, net in routes.items():

        for route in net:
            for seg1, seg2 in zip(route, route[1:]):
                node1 = segment_to_node(seg1, net_id)
                graph.add_node(node1)
                node2 = segment_to_node(seg2, net_id)
                graph.add_node(node2)
                graph.add_edge(node1, node2)

                if node1.route_type == "PORT":
                    tile_id = get_tile_at(
                        node1.x, node1.y, node1.bit_width, placement, node1.port)
                    graph.add_edge(tile_id, node1)
                elif node1.route_type == "REG":
                    tile_id = get_tile_at(
                        node1.x, node1.y, node1.bit_width, placement, reg=True)
                    graph.add_edge(tile_id, node1)

                if node2.route_type == "PORT":
                    tile_id = get_tile_at(
                        node2.x, node2.y, node2.bit_width, placement, node2.port)
                    if tile_id[0] == "m" and "chain" not in node2.port:
                        node2.reg = True
                    graph.add_edge(node2, tile_id)
                elif node2.route_type == "REG":
                    tile_id = get_tile_at(
                        node2.x, node2.y, node2.bit_width, placement, reg=True)
                    node2.reg = True
                    graph.add_edge(node2, tile_id)

    graph.update_sources_and_sinks()
    graph.update_edge_kernels()
    if 'PIPELINED' in os.environ and os.environ['PIPELINED'] == '1':
        for pe in graph.get_ponds()+graph.get_pes():
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
