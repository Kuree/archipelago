import sys
import os
from typing import Dict, List, Set, Union
from enum import Enum


class RouteType(Enum):
    SB = 1
    RMUX = 2
    PORT = 3
    REG = 4


class RouteNode:
    def __init__(
        self,
        x,
        y,
        route_type=None,
        track=None,
        side=None,
        io=None,
        bit_width=None,
        port=None,
        net_id=None,
        reg_name=None,
        rmux_name=None,
        reg=False,
        kernel=None,
    ):

        assert x is not None
        self.x = x
        assert y is not None
        self.y = y

        self.tile_id = (
            f"{route_type or 0},{x or 0},{y or 0},"
            + f"{track or 0},{side or 0},{io or 0},{bit_width or 0},{port or 0},"
            + f"{net_id or 0},{reg_name or 0},{rmux_name or 0},{reg},{kernel}"
        )
        assert self.tile_id is not None

        self.route_type = route_type
        assert self.route_type is not None

        self.track = track
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
        self.tile_id = (
            f"{self.route_type or 0},"
            + f"{self.x or 0},{self.y or 0},{self.track or 0},"
            + f"{self.side or 0},{self.io or 0},"
            + f"{self.bit_width or 0},{self.port or 0},"
            + f"{self.net_id or 0},{self.reg_name or 0},"
            + f"{self.rmux_name or 0},{self.reg},{self.kernel}"
        )
        assert self.tile_id is not None

    def to_route(self):
        if self.route_type == RouteType.SB:
            route_string = [
                "SB",
                self.track,
                self.x,
                self.y,
                self.side,
                self.io,
                self.bit_width,
            ]
        elif self.route_type == RouteType.PORT:
            route_string = ["PORT", self.port, self.x, self.y, self.bit_width]
        elif self.route_type == RouteType.REG:
            route_string = [
                "REG",
                self.reg_name,
                self.track,
                self.x,
                self.y,
                self.bit_width,
            ]
        elif self.route_type == RouteType.RMUX:
            route_string = ["RMUX", self.rmux_name, self.x, self.y, self.bit_width]
        else:
            raise ValueError("Unrecognized route type")
        return route_string

    def __str__(self):
        self.update_tile_id()
        return f"{self.tile_id}"


class TileType(Enum):
    PE = 1
    MEM = 2
    REG = 3
    POND = 4
    IO16 = 5
    IO1 = 6


class TileNode:
    def __init__(self, x, y, tile_id, kernel):

        self.x = x
        self.y = y

        self.tile_id = tile_id

        if self.tile_id[0] == "p":
            self.tile_type = TileType.PE
        elif self.tile_id[0] == "m":
            self.tile_type = TileType.MEM
        elif self.tile_id[0] == "M":
            self.tile_type = TileType.POND
        elif self.tile_id[0] == "r":
            self.tile_type = TileType.REG
        elif self.tile_id[0] == "I":
            self.tile_type = TileType.IO16
        elif self.tile_id[0] == "i":
            self.tile_type = TileType.IO1

        self.kernel = kernel

        self.input_port_latencies = {}
        self.input_port_break_path = {}

    def update_tile_id(self):
        pass

    def __str__(self):
        return f"{self.tile_id}"


class RoutingResultGraph:
    def __init__(self):
        self.nodes: Set[Union[RouteNode, TileNode]] = set()
        self.tile_id_to_tile: Dict[str, Union[RouteNode, TileNode]] = {}
        self.edges: Set[
            (Union[RouteNode, TileNode], Union[RouteNode, TileNode])
        ] = set()
        self.edge_weights: Dict[
            (Union[RouteNode, TileNode], Union[RouteNode, TileNode]), int
        ] = {}
        self.inputs: Set[Union[RouteNode, TileNode]] = set()
        self.outputs: Set[Union[RouteNode, TileNode]] = set()
        self.sources: Dict[
            Union[RouteNode, TileNode], List[Union[RouteNode, TileNode]]
        ] = {}
        self.sinks: Dict[
            Union[RouteNode, TileNode], List[Union[RouteNode, TileNode]]
        ] = {}
        self.placement = {}
        self.id_to_ports = {}
        self.id_to_name: Dict[str, str] = {}
        self.added_regs = 0
        self.mems = None
        self.pes = None
        self.ponds = None
        self.input_ios = None
        self.output_ios = None
        self.regs = None
        self.shift_regs = None
        self.roms = None

    def get_tile(self, tile_id):
        if tile_id in self.tile_id_to_tile:
            return self.tile_id_to_tile[tile_id]
        return None

    def get_tiles(self):
        tiles = []
        for node in self.nodes:
            if isinstance(node, TileNode):
                tiles.append(node)
        return tiles

    def get_routes(self):
        routes = []
        for node in self.nodes:
            if isinstance(node, RouteNode):
                routes.append(node)
        return routes

    def get_mems(self):
        if not self.mems:
            mems = []
            for node in self.nodes:
                if isinstance(node, TileNode) and node.tile_type == TileType.MEM:
                    mems.append(node)
            self.mems = mems
        return self.mems

    def get_roms(self):
        if not self.roms:
            mems = []
            for node in self.nodes:
                if isinstance(node, TileNode) and node.tile_type == TileType.MEM:
                    if "rom_" in self.id_to_name[node.tile_id]:
                        mems.append(node)
            self.roms = mems
        return self.roms

    def get_regs(self):
        if not self.regs:
            regs = []
            for node in self.nodes:
                if isinstance(node, TileNode) and node.tile_type == TileType.REG:
                    regs.append(node)
            self.regs = regs
        return self.regs

    def get_shift_regs(self):
        if not self.shift_regs:
            regs = []
            for node in self.nodes:
                if (
                    isinstance(node, TileNode)
                    and node.tile_type == TileType.REG
                    and "d_reg_" in self.id_to_name[node.tile_id]
                ):
                    regs.append(node)
            self.shift_regs = regs
        return self.shift_regs

    def get_ponds(self):
        if not self.ponds:
            ponds = []
            for node in self.nodes:
                if isinstance(node, TileNode) and node.tile_type == TileType.POND:
                    ponds.append(node)
            self.ponds = ponds
        return self.ponds

    def get_pes(self):
        if not self.pes:
            pes = []
            for node in self.nodes:
                if isinstance(node, TileNode) and node.tile_type == TileType.PE:
                    pes.append(node)
            self.pes = pes
        return self.pes

    def get_input_ios(self):
        if not self.input_ios:
            ios = []
            for node in self.nodes:
                if (
                    isinstance(node, TileNode)
                    and (
                        node.tile_type == TileType.IO16
                        or node.tile_type == TileType.IO1
                    )
                    and len(self.sources[node]) == 0
                ):
                    ios.append(node)
            self.input_ios = ios
        return self.input_ios

    def get_output_ios(self):
        if not self.output_ios:
            ios = []
            for node in self.nodes:
                if (
                    isinstance(node, TileNode)
                    and (
                        node.tile_type == TileType.IO16
                        or node.tile_type == TileType.IO1
                    )
                    and len(self.sinks[node]) == 0
                ):
                    ios.append(node)
            self.output_ios = ios
        return self.output_ios

    def get_inputs_of_kernel(self, kernel):
        kernel_nodes = set()
        for node in self.nodes:
            if node.kernel == kernel:
                kernel_nodes.add(node)

        kernel_input_nodes = set()

        for sink in kernel_nodes:
            visited = set()
            queue = []

            queue.append(sink)
            visited.add(sink)

            while queue:
                n = queue.pop()

                if n.kernel != kernel:
                    kernel_input_nodes.add(sink)
                    break
                elif n != sink:
                    continue

                for node in self.sources[n]:
                    if node not in visited:
                        queue.append(node)
                        visited.add(node)
        if len(kernel_input_nodes) == 0:
            breakpoint()
        return kernel_input_nodes

    def get_outputs_of_kernel(self, kernel):
        kernel_nodes = set()
        for node in self.nodes:
            if node.kernel == kernel:
                kernel_nodes.add(node)

        kernel_output_nodes = set()

        for source in kernel_nodes:
            visited = set()
            queue = []

            queue.append(source)
            visited.add(source)

            while queue:
                n = queue.pop()

                if n.kernel != kernel:
                    kernel_output_nodes.add(source)
                    break
                elif n != source:
                    continue

                for node in self.sinks[n]:
                    if node not in visited:
                        queue.append(node)
                        visited.add(node)
        return kernel_output_nodes

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

    def add_node(self, node):
        if node.tile_id not in self.tile_id_to_tile:
            self.nodes.add(node)
            self.tile_id_to_tile[node.tile_id] = node

    def add_edge(self, node1, node2):
        assert node1 in self.nodes, f"{node1} not in nodes"
        assert node2 in self.nodes, f"{node2} not in nodes"

        assert isinstance(node1, RouteNode) or isinstance(node1, TileNode)
        assert isinstance(node2, RouteNode) or isinstance(node2, TileNode)

        self.edges.add((node1, node2))

        if node2 not in self.sources:
            self.sources[node2] = [node1]

        if node1 not in self.sinks:
            self.sinks[node1] = [node2]

    def update_sources_and_sinks(self):
        self.inputs = set()
        self.outputs = set()
        self.sources = {}
        self.sinks = {}

        for node in self.nodes:
            self.sources[node] = []
            self.sinks[node] = []

        for source, sink in self.edges:
            assert source in self.nodes
            assert sink in self.nodes
            assert isinstance(source, RouteNode) or isinstance(source, TileNode)
            assert isinstance(sink, RouteNode) or isinstance(sink, TileNode)
            self.sources[sink].append(source)
            self.sinks[source].append(sink)

        for node in self.nodes:
            if len(self.sources[node]) == 0:
                self.inputs.add(node)
            if len(self.sinks[node]) == 0:
                self.outputs.add(node)

    def topological_sort(self):
        sys.setrecursionlimit(10**6)
        visited = set()
        stack = []
        for n in self.inputs:
            if n not in visited:
                self.topological_sort_helper(n, stack, visited)
        return stack[::-1]

    def topological_sort_helper(self, node, stack, visited):
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
        sys.setrecursionlimit(10**6)
        visited = []
        rec_stack = []
        for node in self.inputs:
            if node not in visited:
                break_edge = self.is_cyclic_util(node, visited, rec_stack)
                if break_edge is not None:
                    self.remove_edge(break_edge)
                    return True
        return False

    def segment_to_node(self, segment, net_id, kernel=None):
        if segment[0] == "SB":
            track, x, y, side, io_, bit_width = segment[1:]
            node = RouteNode(
                x,
                y,
                route_type=RouteType.SB,
                track=track,
                side=side,
                io=io_,
                bit_width=bit_width,
                net_id=net_id,
                kernel=kernel,
            )
        elif segment[0] == "PORT":
            port_name, x, y, bit_width = segment[1:]
            node = RouteNode(
                x,
                y,
                route_type=RouteType.PORT,
                bit_width=bit_width,
                net_id=net_id,
                port=port_name,
                kernel=kernel,
            )
        elif segment[0] == "REG":
            reg_name, track, x, y, bit_width = segment[1:]
            node = RouteNode(
                x,
                y,
                route_type=RouteType.REG,
                track=track,
                bit_width=bit_width,
                net_id=net_id,
                reg_name=reg_name,
                port="reg",
                kernel=kernel,
            )
        elif segment[0] == "RMUX":
            rmux_name, x, y, bit_width = segment[1:]
            node = RouteNode(
                x,
                y,
                route_type=RouteType.RMUX,
                bit_width=bit_width,
                net_id=net_id,
                rmux_name=rmux_name,
                kernel=kernel,
            )
        else:
            raise ValueError("Unrecognized route type")

        if node.tile_id in self.tile_id_to_tile:
            return self.tile_id_to_tile[node.tile_id]
        return node

    def gen_placement(self, placement, netlist, pes_with_packed_ponds):
        for blk_id, place in placement.items():
            if place not in self.placement:
                self.placement[place] = []
            self.placement[place].append(blk_id)

        for net_id, conns in netlist.items():
            for conn in conns:
                if (
                    pes_with_packed_ponds is not None
                    and conn[0] in pes_with_packed_ponds
                    and "PondTop" in conn[1]
                ):
                    new_conn = pes_with_packed_ponds[conn[0]]
                else:
                    new_conn = conn[0]

                if new_conn not in self.id_to_ports:
                    self.id_to_ports[new_conn] = []
                self.id_to_ports[new_conn].append(conn[1])

    def get_tile_at(self, x, y, port, pes_with_packed_ponds):
        tiles = self.placement[(x, y)]

        for tile in tiles:
            if pes_with_packed_ponds is not None and tile in pes_with_packed_ponds:
                if port in self.id_to_ports[pes_with_packed_ponds[tile]]:
                    return pes_with_packed_ponds[tile]
            if port in self.id_to_ports[tile]:
                return tile

        return None

    def get_or_create_reg_at(self, x, y, track, bit_width, reg_name):
        tiles = self.get_tiles()

        for tile in tiles:
            if (
                tile.tile_type == TileType.REG
                and tile.x == x
                and tile.y == y
                and tile.track == track
                and tile.bit_width == bit_width
                and tile.reg_name == reg_name
            ):
                return tile

        node = TileNode(x, y, tile_id=f"r{self.added_regs}", kernel=None)
        node.track = track
        node.bit_width = bit_width
        node.reg_name = reg_name
        self.add_node(node)

        self.id_to_name[node.tile_id] = f"pnr_pipelining{self.added_regs}"

        self.added_regs += 1

        return node

    def update_edge_kernels(self):
        nodes = self.topological_sort()

        for in_node in nodes:
            assert in_node.kernel is not None
            for node in self.sinks[in_node]:
                if isinstance(node, RouteNode) or (
                    node.tile_type == TileType.REG and node.kernel is None
                ):
                    node.kernel = in_node.kernel
                else:
                    assert node.kernel is not None

        for tile in self.get_tiles():
            assert tile.kernel is not None, tile
            for source in self.sources[tile]:
                source.kernel = tile.kernel

        for node in self.nodes:
            node.update_tile_id()
            assert node.kernel is not None, node

    def fix_regs(self, netlist):
        for tile in self.get_tiles():
            if tile.tile_type == TileType.REG:
                if len(self.sinks[tile]) == 0:
                    # If one isn't hooked up correctly we need to fix it
                    # Pretty hacky but works
                    source = self.sources[tile][0]
                    source_copy = RouteNode(
                        source.x,
                        source.y,
                        route_type=RouteType.REG,
                        track=source.track,
                        bit_width=source.bit_width,
                        net_id=source.net_id,
                        reg_name=source.reg_name,
                        port="reg",
                        kernel=source.kernel,
                    )

                    source_copy.reg = True
                    source_copy.update_tile_id()
                    self.add_node(source_copy)
                    self.add_edge(tile, source_copy)

                    for source_sink in self.sinks[source]:
                        if source_sink != tile:
                            self.remove_edge((source, source_sink))
                            self.add_edge(source_copy, source_sink)

        # Routing result doesn't have reg name information
        # Need to get that from the netlist
        unsolved_regs = []
        for node in self.get_tiles():
            if node.tile_type == TileType.REG:
                assert (
                    node.x,
                    node.y,
                ) in self.placement, (
                    f"Reg at ({node.x},{node.y}) not in placement result"
                )
                regs = self.placement[(node.x, node.y)]
                regs = [reg for reg in regs if reg[0] == "r"]
                if len(regs) == 1:
                    node.tile_id = regs[0]

                    if len(self.id_to_name[regs[0]].split("$")) > 0:
                        kernel = self.id_to_name[regs[0]].split("$")[0]
                    else:
                        kernel = None

                    node.kernel = kernel

                else:
                    unsolved_regs.append((node, regs))

        seen_regs = set()

        while len(unsolved_regs) > 0:
            resolved = False
            (node, regs) = unsolved_regs.pop(0)
            if node in seen_regs:
                print(f"Couldn't associate {node} with {regs} in placement file")
                return
            seen_regs.add(node)

            prev_tile_found = False
            prev_node = node
            while not prev_tile_found:
                assert len(self.sources[prev_node]) == 1
                prev_node = self.sources[prev_node][0]

                if isinstance(prev_node, TileNode):
                    prev_tile_found = True

            if prev_node.kernel != None:
                for net_id, net in netlist.items():
                    if net[0][0] == prev_node.tile_id:
                        for id_ in net[1:]:
                            if id_[0] in regs:
                                resolved = True
                                node.tile_id = id_[0]
                                node.kernel = self.id_to_name[node.tile_id].split("$")[
                                    0
                                ]
                                seen_regs = set()
            if not resolved:
                unsolved_regs.append((node, regs))

    def get_output_tiles_of_kernel(self, kernel):
        kernel_nodes = set()
        for node in self.nodes:
            if node.kernel == kernel:
                kernel_nodes.add(node)

        kernel_output_nodes = set()

        for source in kernel_nodes:
            visited = set()
            queue = []

            queue.append(source)
            visited.add(source)

            while queue:
                n = queue.pop()

                if isinstance(n, TileNode) and n.kernel != kernel:
                    kernel_output_nodes.add(source)
                    break
                elif n != source and isinstance(n, TileNode):
                    continue

                for node in self.sinks[n]:
                    if node not in visited:
                        queue.append(node)
                        visited.add(node)
        return kernel_output_nodes


def construct_graph(
    placement,
    routes,
    id_to_name,
    netlist,
    pe_latency=0,
    pond_latency=0,
    io_latency=0,
    sparse=False,
    pes_with_packed_ponds=None,
):
    graph = RoutingResultGraph()
    graph.id_to_name = id_to_name
    graph.sparse = sparse
    graph.gen_placement(placement, netlist, pes_with_packed_ponds)

    max_reg_id = 0

    for blk_id, place in placement.items():
        if blk_id[0] != "r":
            if len(graph.id_to_name[blk_id].split("$")) > 0:
                kernel = graph.id_to_name[blk_id].split("$")[0]
            else:
                kernel = None
            node = TileNode(place[0], place[1], tile_id=blk_id, kernel=kernel)
            graph.add_node(node)

            if pes_with_packed_ponds is not None and blk_id in pes_with_packed_ponds:
                pond_id = pes_with_packed_ponds[blk_id]
                node = TileNode(place[0], place[1], tile_id=pond_id, kernel=kernel)
                graph.add_node(node)

        max_reg_id = max(max_reg_id, int(blk_id[1:]))
    graph.added_regs = max_reg_id + 1

    for net_id, net in routes.items():
        for route in net:
            for seg1, seg2 in zip(route, route[1:]):
                node1 = graph.segment_to_node(seg1, net_id)
                graph.add_node(node1)
                node2 = graph.segment_to_node(seg2, net_id)
                graph.add_node(node2)
                graph.add_edge(node1, node2)

                if node1.route_type == RouteType.PORT:
                    tile_id = graph.get_tile_at(
                        node1.x, node1.y, node1.port, pes_with_packed_ponds
                    )
                    graph.add_edge(graph.get_tile(tile_id), node1)
                elif node1.route_type == RouteType.REG:
                    reg_tile = graph.get_or_create_reg_at(
                        node1.x,
                        node1.y,
                        node1.track,
                        node1.bit_width,
                        node1.reg_name,
                    )
                    graph.add_edge(reg_tile, node1)

                if node2.route_type == RouteType.PORT:
                    tile_id = graph.get_tile_at(
                        node2.x, node2.y, node2.port, pes_with_packed_ponds
                    )
                    graph.add_edge(node2, graph.get_tile(tile_id))
                elif node2.route_type == RouteType.REG:
                    reg_tile = graph.get_or_create_reg_at(
                        node2.x,
                        node2.y,
                        node2.track,
                        node2.bit_width,
                        node2.reg_name,
                    )
                    graph.add_edge(node2, reg_tile)

    graph.update_sources_and_sinks()

    while graph.fix_cycles():
        pass

    graph.fix_regs(netlist)

    id_to_input_ports = {}
    for net_id, conns in netlist.items():
        for conn in conns[1:]:
            if (
                pes_with_packed_ponds is not None
                and conn[0] in pes_with_packed_ponds
                and "PondTop" in conn[1]
            ):
                new_conn = pes_with_packed_ponds[conn[0]]
            else:
                new_conn = conn[0]

            if new_conn not in id_to_input_ports:
                id_to_input_ports[new_conn] = []
            id_to_input_ports[new_conn].append(conn[1])

    for tile in graph.get_tiles():
        tile_id = tile.tile_id
        if tile_id in id_to_input_ports:
            for port in id_to_input_ports[tile_id]:
                if tile.tile_type == TileType.PE:
                    tile.input_port_latencies[port] = pe_latency
                    tile.input_port_break_path[port] = pe_latency != 0
                elif tile.tile_type == TileType.MEM:
                    if "rom_" in id_to_name[tile_id]:
                        tile.input_port_latencies[port] = 1
                        tile.input_port_break_path[port] = True
                    elif "flush" in port:
                        tile.input_port_latencies[port] = 0
                        tile.input_port_break_path[port] = False
                    else:
                        tile.input_port_latencies[port] = 0
                        tile.input_port_break_path[port] = True
                elif tile.tile_type == TileType.REG:
                    if tile in graph.get_shift_regs():
                        tile.input_port_latencies[port] = 0
                        tile.input_port_break_path[port] = True
                    else:
                        tile.input_port_latencies[port] = 1
                        tile.input_port_break_path[port] = True
                elif tile.tile_type == TileType.POND:
                    tile.input_port_latencies[port] = pond_latency
                    tile.input_port_break_path[port] = True
                elif tile.tile_type == TileType.IO1 or tile.tile_type == TileType.IO16:
                    tile.input_port_latencies[port] = 1
                    tile.input_port_break_path[port] = True
        else:
            if tile_id[0] == "r":
                tile.input_port_latencies["reg"] = 1
                tile.input_port_break_path["reg"] = True

    # Need special case for input IO tiles since they don't have an "input" port
    for tile in graph.get_input_ios():
        tile.input_port_latencies["output"] = io_latency
        tile.input_port_break_path["output"] = io_latency != 0

    graph.update_sources_and_sinks()
    graph.update_edge_kernels()

    while graph.fix_cycles():
        pass

    return graph


class KernelNodeType(Enum):
    COMPUTE = 1
    MEM = 2
    RESET = 3
    IO = 4


class KernelNode:
    def __init__(
        self,
        mem_id=None,
        kernel=None,
        kernel_type=None,
        latency=0,
        flush_latency=0,
        has_shift_regs=False,
    ):
        self.mem_id = mem_id
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.latency = latency
        self.flush_latency = flush_latency
        self.has_shift_regs = has_shift_regs

    def __str__(self):
        if self.kernel:
            return f"{self.kernel}"
        else:
            return f"{self.mem_id}"


class KernelGraph:
    def __init__(self):
        self.nodes: Set[KernelNode] = set()
        self.edges: Set[(KernelNode, KernelNode)] = set()
        self.inputs: Set[KernelNode] = set()
        self.outputs: Set[KernelNode] = set()
        self.sources: Dict[KernelNode, List[KernelNode]] = {}
        self.sinks: Dict[KernelNode, List[KernelNode]] = {}
        self.tile_id_to_tile: Dict[str, KernelNode] = {}

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
        if str(node) not in self.tile_id_to_tile:
            self.nodes.add(node)
            self.tile_id_to_tile[str(node)] = node

    def add_edge(self, node1, node2):
        assert node1 in self.nodes, f"{node1} not in nodes"
        assert node2 in self.nodes, f"{node2} not in nodes"

        assert isinstance(node1, KernelNode)
        assert isinstance(node2, KernelNode)

        self.edges.add((node1, node2))

        if node2 not in self.sources:
            self.sources[node2] = []
        self.sources[node2].append(node1)

        if node1 not in self.sinks:
            self.sinks[node1] = []
        self.sinks[node1].append(node2)

    def update_sources_and_sinks(self):
        self.inputs = set()
        self.outputs = set()
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
                self.inputs.add(node)
            if len(self.sinks[node]) == 0:
                self.outputs.add(node)

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


def construct_kernel_graph(graph, new_latencies):
    kernel_graph = KernelGraph()

    compute_tiles = set()
    for tile in (
        graph.get_pes() + graph.get_regs() + graph.get_ponds() + graph.get_input_ios()
    ):
        if tile not in graph.get_shift_regs():
            compute_tiles.add(tile)

    for source in graph.get_tiles():
        if source in compute_tiles:
            source_id = source.kernel
            if source_id not in kernel_graph.tile_id_to_tile:
                kernel_node = KernelNode(kernel=source_id)
                kernel_graph.add_node(kernel_node)
                kernel_node.latency = new_latencies[source_id]
                kernel_node.kernel_type = KernelNodeType.COMPUTE
        else:
            source_id = source.tile_id
            if source_id not in kernel_graph.tile_id_to_tile:
                kernel_node = KernelNode(kernel=source_id)
                kernel_graph.add_node(kernel_node)
                kernel_node.kernel_type = KernelNodeType.MEM
                if "reset" in source.kernel:
                    kernel_node.kernel_type = KernelNodeType.RESET

        for dest in graph.get_tiles():
            if dest in compute_tiles:
                dest_id = dest.kernel
                if dest_id not in kernel_graph.tile_id_to_tile:
                    kernel_node = KernelNode(kernel=dest_id)
                    kernel_graph.add_node(kernel_node)
                    kernel_node.latency = new_latencies[dest_id]
                    kernel_node.kernel_type = KernelNodeType.COMPUTE
            else:
                dest_id = dest.tile_id
                if dest_id not in kernel_graph.tile_id_to_tile:
                    kernel_node = KernelNode(kernel=dest_id)
                    kernel_graph.add_node(kernel_node)
                    kernel_node.kernel_type = KernelNodeType.MEM
                    if "reset" in dest.kernel:
                        kernel_node.kernel_type = KernelNodeType.RESET

            if str(source) != str(dest):
                reachable = False
                visited = set()
                queue = []
                queue.append(source)
                visited.add(source)
                while queue:
                    n = queue.pop()

                    if str(n) == str(dest) and str(n) != str(source):
                        reachable = True

                    for node in graph.sinks[n]:
                        if node not in visited:
                            if isinstance(node, TileNode):
                                if str(node) == str(dest) and str(node) != str(source):
                                    reachable = True
                            else:
                                queue.append(node)
                                visited.add(node)

                if reachable and source_id != dest_id:
                    kernel_graph.add_edge(
                        kernel_graph.tile_id_to_tile[source_id],
                        kernel_graph.tile_id_to_tile[dest_id],
                    )

    kernel_graph.update_sources_and_sinks()

    return kernel_graph
