import os
import copy
import json
import argparse
import sys
from pycyclone.io import load_placement
import pycyclone
import pythunder
from archipelago.io import load_routing_result
from archipelago.pnr_graph import (
    RoutingResultGraph,
    construct_graph,
    TileType,
    RouteType,
    TileNode,
    RouteNode,
)
from archipelago.visualize import visualize_pnr


class PathComponents:
    def __init__(
        self,
        glbs=0,
        sb_delay=[],
        sb_clk_delay=[],
        pes=0,
        mems=0,
        rmux=0,
        available_regs=0,
        parent=None,
    ):
        self.glbs = glbs
        self.sb_delay = sb_delay
        self.sb_clk_delay = sb_clk_delay
        self.pes = pes
        self.mems = mems
        self.rmux = rmux
        self.available_regs = available_regs
        self.parent = parent
        self.delays = json.load(
            open(os.path.dirname(os.path.realpath(__file__)) + "/sta_delays.json")
        )

    def get_total(self):
        total = 0
        total += self.glbs * self.delays["glb"]
        total += self.pes * self.delays["pe"]
        total += self.mems * self.delays["mem"]
        total += self.rmux * self.delays["rmux"]
        total += sum(self.sb_delay)
        total -= sum(self.sb_clk_delay)
        return total

    def print(self):
        print("\t\tGlbs:", self.glbs)
        print("\t\tPEs:", self.pes)
        print("\t\tMems:", self.mems)
        print("\t\tRmux:", self.rmux)
        print("\t\tSB delay:", sum(self.sb_delay), "ps")
        print("\t\tSB delay:", self.sb_delay, "ps")
        print("\t\tSB clk delay:", sum(self.sb_clk_delay), "ps")
        print("\t\tSB clk delay:", self.sb_clk_delay, "ps")


def get_mem_tile_columns(graph):
    mem_column = 4
    for mem in graph.get_mems():
        if (mem.x + 1) % mem_column != 0:
            raise ValueError("MEM tile not at expected column, please update me")

    return mem_column




def calc_sb_delay(graph, node, parent, comp, mem_column, sparse):
    # Need to associate each sb hop with these catagories:
    # mem2pe_clk
    # pe2mem_clk
    # north_input_clk
    # south_input_clk
    # pe2pe_west_east_input_clk
    # mem_endpoint_sb
    # pe_endpoint_sb

    if graph.sinks[node][0].route_type == RouteType.PORT:
        if graph.sinks[graph.sinks[node][0]]:
            if graph.sinks[graph.sinks[node][0]][0].tile_type == TileType.MEM:
                comp.sb_delay.append(comp.delays[f"SB_IN_to_MEM"])
            else:
                comp.sb_delay.append(comp.delays[f"SB_IN_to_PE"])

    if parent.io == 0:
        # Its the input to the SB
        if parent.side == 0:
            # Coming in from right
            source_x = parent.x + 1
        elif parent.side == 1:
            # Coming in from bottom
            source_x = parent.x
        elif parent.side == 2:
            # Coming in from left
            source_x = parent.x - 1
        else:
            # Coming in from top
            source_x = parent.x
        next_sb = node
        if next_sb.route_type != RouteType.SB:
            return
        assert next_sb.io == 1
        source_mem = False
        if (source_x + 1) % mem_column == 0:
            # Starting at mem column
            source_mem = True

        dest_mem = False
        if (next_sb.x + 1) % mem_column == 0:
            # Starting at mem column
            dest_mem = True

        if source_mem and not dest_mem:
            # mem2pe_clk
            comp.sb_clk_delay.append(comp.delays["mem2pe_clk"])
        elif not source_mem and dest_mem:
            # pe2mem_clk
            comp.sb_clk_delay.append(comp.delays["pe2mem_clk"])
        elif parent.side == 3:
            # north_input_clk
            comp.sb_clk_delay.append(comp.delays["north_input_clk"])
        elif parent.side == 1:
            # south_input_clk
            comp.sb_clk_delay.append(comp.delays["south_input_clk"])
        else:
            # pe2pe_west_east_input_clk
            comp.sb_clk_delay.append(comp.delays["pe2pe_west_east_input_clk"])

        side_to_dir = {0: "EAST", 1: "SOUTH", 2: "WEST", 3: "NORTH"}

        if not sparse:
            if (parent.x + 1) % mem_column == 0:
                comp.sb_delay.append(
                    comp.delays[
                        f"MEM_B{parent.bit_width}_{side_to_dir[parent.side]}_{side_to_dir[next_sb.side]}"
                    ]
                )
            else:
                comp.sb_delay.append(
                    comp.delays[
                        f"PE_B{parent.bit_width}_{side_to_dir[parent.side]}_{side_to_dir[next_sb.side]}"
                    ]
                )
        else:

            if (parent.x + 1) % mem_column == 0:
                comp.sb_delay.append(
                    comp.delays[
                        f"MEM_B{parent.bit_width}_valid_{side_to_dir[parent.side]}_{side_to_dir[next_sb.side]}"
                    ]
                )
            else:
                comp.sb_delay.append(
                    comp.delays[
                        f"PE_B{parent.bit_width}_valid_{side_to_dir[parent.side]}_{side_to_dir[next_sb.side]}"
                    ]
                )

            if (parent.x + 1) % mem_column == 0:
                comp.sb_delay.append(
                    comp.delays[
                        f"MEM_B{parent.bit_width}_ready_{side_to_dir[next_sb.side]}_{side_to_dir[parent.side]}"
                    ]
                )
            else:
                comp.sb_delay.append(
                    comp.delays[
                        f"PE_B{parent.bit_width}_ready_{side_to_dir[next_sb.side]}_{side_to_dir[parent.side]}"
                    ]
                )


def sta(graph):

    mem_tile_column = get_mem_tile_columns(graph)
    nodes = graph.topological_sort()
    timing_info = {}

    for node in nodes:
        comp = PathComponents()
        components = [comp]

        if len(graph.sources[node]) == 0 and (
            node.tile_type == TileType.IO16 or node.tile_type == TileType.IO1
        ):
            if not node.input_port_break_path["output"]:
                comp = PathComponents()
                comp.glbs = 1
                components = [comp]

        for parent in graph.sources[node]:
            comp = PathComponents()

            if parent in timing_info:
                comp = copy.deepcopy(timing_info[parent])
                comp.parent = parent

            if isinstance(node, TileNode):
                if node.tile_type == TileType.PE:
                    comp.pes += 1
                elif node.tile_type == TileType.MEM:
                    comp.mems += 1
                elif node.tile_type == TileType.IO16 or node.tile_type == TileType.IO1:
                    comp.glbs += 1
            else:
                if len(graph.sinks[node]) == 0:
                    continue
                if node.route_type == RouteType.PORT and isinstance(
                    graph.sinks[node][0], TileNode
                ):
                    if graph.sinks[node][0].input_port_break_path[node.port]:
                        comp = PathComponents()
                elif node.route_type == RouteType.REG and isinstance(
                    graph.sinks[node][0], TileNode
                ):
                    # if graph.sinks[node][0].input_port_break_path["reg"]:
                    comp = PathComponents()
                elif node.route_type == RouteType.SB:
                    calc_sb_delay(
                        graph, node, parent, comp, mem_tile_column, graph.sparse
                    )
                elif node.route_type == RouteType.RMUX:
                    if (
                        isinstance(parent, RouteNode)
                        and parent.route_type == RouteType.REG
                    ):
                        comp.rmux += 1
                    if parent.route_type != RouteType.REG:
                        comp.available_regs += 1

            components.append(comp)

        maxt = 0
        max_comp = components[0]
        for comp in components:
            if comp.get_total() > maxt:
                maxt = comp.get_total()
                max_comp = comp

        timing_info[node] = max_comp

    node_to_timing = {node: timing_info[node].get_total() for node in graph.nodes}
    node_to_timing = dict(
        sorted(
            reversed(list(node_to_timing.items())),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    max_node = list(node_to_timing.keys())[0]
    max_delay = list(node_to_timing.values())[0]

    clock_speed = int(1.0e12 / max_delay / 1e6)

    print("\tMaximum clock frequency:", clock_speed, "MHz")
    print("\tCritical Path:", max_delay, "ps")
    print("\tCritical Path Info:")
    timing_info[max_node].print()

    max_node = list(node_to_timing.keys())[0]
    curr_node = max_node
    crit_path = []
    crit_path.append((curr_node, timing_info[curr_node].get_total()))
    crit_nodes = []
    while True:
        crit_nodes.append(curr_node)
        curr_node = timing_info[curr_node].parent
        crit_path.append((curr_node, timing_info[curr_node].get_total()))
        if timing_info[curr_node].parent is None:
            break

    crit_path.reverse()

    return clock_speed, crit_path, crit_nodes


def load_id_to_name(id_filename):
    fin = open(id_filename, "r")
    lines = fin.readlines()
    id_to_name = {}

    for line in lines:
        id_to_name[line.split(": ")[0]] = line.split(": ")[1].rstrip()

    return id_to_name


def load_graph(graph_files):
    graph_result = {}
    for graph_file in graph_files:
        bit_width = os.path.splitext(graph_file)[0]
        bit_width = int(os.path.basename(bit_width))
        graph = pycyclone.io.load_routing_graph(graph_file)
        graph_result[bit_width] = graph
    return graph_result


def parse_args():
    parser = argparse.ArgumentParser("CGRA timing analysis tool")
    parser.add_argument(
        "-a", "--app", "-d", required=True, dest="application", type=str
    )
    parser.add_argument("-v", "--visualize", action="store_true")
    parser.add_argument("-s", "--sparse", action="store_true")
    args = parser.parse_args()
    dirname = args.application  # os.path.join(args.application, "bin")
    netlist = os.path.join(dirname, "design.packed")
    assert os.path.exists(netlist), netlist + " does not exist"
    placement = os.path.join(dirname, "design.place")
    assert os.path.exists(placement), placement + " does not exists"
    route = os.path.join(dirname, "design.route")
    assert os.path.exists(route), route + " does not exists"
    id_to_name_filename = os.path.join(dirname, "design.id_to_name")
    return netlist, placement, route, id_to_name_filename, args.visualize, args.sparse


def run_sta(packed_file, placement_file, routing_file, id_to_name, sparse):

    netlist, buses = pythunder.io.load_netlist(packed_file)
    placement = load_placement(placement_file)
    routing = load_routing_result(routing_file)

    if "PIPELINED" in os.environ and os.environ["PIPELINED"].isnumeric():
        pe_latency = int(os.environ["PIPELINED"])
    else:
        pe_latency = 1

    if "IO_DELAY" in os.environ and os.environ["IO_DELAY"] == "0":
        io_cycles = 0
    else:
        io_cycles = 1

    routing_result_graph = construct_graph(
        placement, routing, id_to_name, netlist, pe_latency, 0, io_cycles, sparse
    )

    break_at_mems(routing_result_graph, id_to_name, placement, routing, sparse)

    clock_speed, crit_path, crit_nodes = sta(routing_result_graph)

    return clock_speed

def find_break_idx(graph, crit_path):
    crit_path_adjusted = [abs(c - crit_path[-1][1] / 2) for n, c in crit_path]
    break_idx = crit_path_adjusted.index(min(crit_path_adjusted))

    if len(crit_path) < 2:
        raise ValueError("Can't find available register on critical path")

    if graph.sparse and len(crit_path) < 5:
        raise ValueError("Can't find available FIFO on critical path")

    min_path = crit_path[-1][1]
    min_idx = -1

    if graph.sparse:
        for idx, node in enumerate(crit_path[:-4]):
            if (
                isinstance(crit_path[idx][0], RouteNode)
                and crit_path[idx][0].route_type == RouteType.SB
                and isinstance(crit_path[idx + 1][0], RouteNode)
                and crit_path[idx + 1][0].route_type == RouteType.RMUX
                and isinstance(crit_path[idx + 2][0], RouteNode)
                and crit_path[idx + 2][0].route_type == RouteType.SB
                and isinstance(crit_path[idx + 3][0], RouteNode)
                and crit_path[idx + 3][0].route_type == RouteType.SB
                and isinstance(crit_path[idx + 4][0], RouteNode)
                and crit_path[idx + 4][0].route_type == RouteType.RMUX
            ):
                if crit_path_adjusted[idx] < min_path:
                    min_path = crit_path_adjusted[idx]
                    min_idx = idx
    else:
        for idx, node in enumerate(crit_path[:-1]):
            if (
                isinstance(crit_path[idx][0], RouteNode)
                and crit_path[idx][0].route_type == RouteType.SB
                and isinstance(crit_path[idx + 1][0], RouteNode)
                and crit_path[idx + 1][0].route_type == RouteType.RMUX
            ):
                if crit_path_adjusted[idx] < min_path:
                    min_path = crit_path_adjusted[idx]
                    min_idx = idx

    if min_idx == -1:
        raise ValueError("Can't find available register on critical path")

    return min_idx


def reg_into_route(routes, g_break_node_source, new_reg_route_source):
    for net_id, net in routes.items():
        for route in net:
            for idx, segment in enumerate(route):
                if g_break_node_source.to_route() == segment:
                    route.insert(idx + 1, new_reg_route_source.to_route())
                    return
    assert (
        False
    ), f"Couldn't find segment {g_break_node_source.to_route()} in routing file"


def break_crit_path(graph, id_to_name, crit_path, placement, routes):
    break_idx = find_break_idx(graph, crit_path)

    break_node_source = crit_path[break_idx][0]
    break_node_dest = graph.sinks[break_node_source][0]

    assert isinstance(break_node_source, RouteNode)
    assert break_node_source.route_type == RouteType.SB
    assert isinstance(break_node_dest, RouteNode)
    assert break_node_dest.route_type == RouteType.RMUX

    x = break_node_source.x
    y = break_node_source.y
    track = break_node_source.track
    bw = break_node_source.bit_width
    net_id = break_node_source.net_id
    kernel = break_node_source.kernel
    side = break_node_source.side
    print("\nBreaking net:", net_id, "Kernel:", kernel)

    dir_map = {0: "EAST", 1: "SOUTH", 2: "WEST", 3: "NORTH"}

    new_segment = ["REG", f"T{track}_{dir_map[side]}", track, x, y, bw]
    new_reg_route_source = graph.segment_to_node(new_segment, net_id, kernel)
    new_reg_route_source.reg = True
    new_reg_route_source.update_tile_id()
    new_reg_route_dest = graph.segment_to_node(new_segment, net_id, kernel)
    new_reg_tile = TileNode(x, y, tile_id=f"r{graph.added_regs}", kernel=kernel)

    new_reg_tile.input_port_latencies["reg"] = 1
    new_reg_tile.input_port_break_path["reg"] = True

    graph.added_regs += 1

    graph.edges.remove((break_node_source, break_node_dest))
    graph.add_node(new_reg_route_source)
    graph.add_node(new_reg_tile)
    graph.add_node(new_reg_route_dest)

    graph.add_edge(break_node_source, new_reg_route_source)
    graph.add_edge(new_reg_route_source, new_reg_tile)
    graph.add_edge(new_reg_tile, new_reg_route_dest)
    graph.add_edge(new_reg_route_dest, break_node_dest)

    reg_into_route(routes, break_node_source, new_reg_route_source)
    placement[new_reg_tile.tile_id] = (new_reg_tile.x, new_reg_tile.y)
    id_to_name[new_reg_tile.tile_id] = f"pnr_pipelining{graph.added_regs}"

    graph.update_sources_and_sinks()
    graph.update_edge_kernels()


def break_at_mem_node(graph, id_to_name, placement, routes, node):
    found = False
    curr_node = node
    while len(graph.sinks[curr_node]) == 1 and not found:
        next_node = graph.sinks[curr_node][0]

        if (
            isinstance(curr_node, RouteNode)
            and curr_node.route_type == RouteType.SB
            and isinstance(next_node, RouteNode)
            and next_node.route_type == RouteType.RMUX
        ):
            crit_path = [(curr_node, 0), (next_node, 1)]
            break_crit_path(graph, id_to_name, crit_path, placement, routes)
            reg = graph.sinks[graph.sinks[curr_node][0]][0]
            reg.input_port_latencies["reg"] = 0
            reg.input_port_break_path["reg"] = True
            found = True

        curr_node = next_node

    if not found:
        found = True
        for sink in graph.sinks[curr_node]:
            found = found & break_at_mem_node(
                graph, id_to_name, placement, routes, sink
            )

    return found


def break_at_mems(graph, id_to_name, placement, routes, sparse):
    if sparse:
        return
    for mem in graph.get_mems():
        for port in graph.sinks[mem]:
            # if str(mem) in chained_mems and port.port == chained_mems[str(mem)]:
            #     print(mem, port.port, "is used in chain mode, skipping mem register")
            #     continue

            found = break_at_mem_node(graph, id_to_name, placement, routes, port)
            assert found, f"Couldn't insert register at output port {port} of {mem}"


def main():
    (
        packed_file,
        placement_file,
        routing_file,
        id_to_name_filename,
        visualize,
        sparse,
    ) = parse_args()

    netlist, buses = pythunder.io.load_netlist(packed_file)

    if os.path.isfile(id_to_name_filename):
        id_to_name = load_id_to_name(id_to_name_filename)
    else:
        id_to_name = pythunder.io.load_id_to_name(packed_file)

    placement = load_placement(placement_file)
    routing = load_routing_result(routing_file)

    if "PIPELINED" in os.environ and os.environ["PIPELINED"].isnumeric():
        pe_latency = int(os.environ["PIPELINED"])
    else:
        pe_latency = 1

    if "IO_DELAY" in os.environ and os.environ["IO_DELAY"] == "0":
        io_cycles = 0
    else:
        io_cycles = 1

    routing_result_graph = construct_graph(
        placement, routing, id_to_name, netlist, pe_latency, 0, io_cycles, sparse
    )

    break_at_mems(routing_result_graph, id_to_name, placement, routing, sparse)

    clock_speed, crit_path, crit_nodes = sta(routing_result_graph)

    if visualize:
        dirname = os.path.dirname(packed_file)
        graph1 = os.path.join(dirname, "1.graph")
        assert os.path.exists(graph1), route + " does not exists"
        graph16 = os.path.join(dirname, "16.graph")
        if not os.path.exists(graph16):
            graph16 = os.path.join(dirname, "17.graph")
        assert os.path.exists(graph16), route + " does not exists"
        routing_graphs = load_graph([graph1, graph16])

        visualize_pnr(routing_graphs, routing_result_graph, crit_nodes, dirname)

    return clock_speed


if __name__ == "__main__":
    main()
