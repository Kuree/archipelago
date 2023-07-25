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
        sb_delay_rv=[],
        sb_clk_delay=[],
        pes=0,
        mems=0,
        rmux=0,
        available_regs=0,
        parent=None,
    ):
        self.glbs = glbs
        self.sb_delay = sb_delay
        self.sb_delay_rv = sb_delay_rv
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
        total_dense = 0
        total_dense += self.glbs * self.delays["glb"]
        total_dense += self.pes * self.delays["pe"]
        total_dense += self.mems * self.delays["mem"]
        total_dense += self.rmux * self.delays["rmux"]
        total_dense += sum(self.sb_delay)
        total_dense -= sum(self.sb_clk_delay)

        total_rv = 0
        total_rv += self.glbs * self.delays["glb"]
        total_rv += self.rmux * self.delays["rmux"]
        total_rv += sum(self.sb_delay_rv)
        total_rv -= sum(self.sb_clk_delay)
        return max(total_dense, total_rv)

    def print(self):
        print("\t\tGlbs:", self.glbs)
        print("\t\tPEs:", self.pes)
        print("\t\tMems:", self.mems)
        print("\t\tRmux:", self.rmux)
        print("\t\tSB delay:", sum(self.sb_delay), "ps")
        print("\t\tSB delay:", self.sb_delay, "ps")
        print("\t\tSB delay ready-valid:", sum(self.sb_delay_rv), "ps")
        print("\t\tSB delay ready-valid:", self.sb_delay_rv, "ps")
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

    if sparse:
        if graph.sinks[node][0].route_type == RouteType.PORT:
            if "MEM" in graph.sinks[node][0].port:
                comp.sb_delay.append(comp.delays[f"SB_IN_to_MEM_fifo"])
                comp.sb_delay_rv.append(comp.delays[f"SB_IN_to_MEM_fifo_valid"])
            else:
                comp.sb_delay.append(comp.delays[f"SB_IN_to_PE_fifo"])
                comp.sb_delay_rv.append(comp.delays[f"SB_IN_to_PE_fifo_valid"])
    else:
        if graph.sinks[node][0].route_type == RouteType.PORT:
            if "MEM" in graph.sinks[node][0].port:
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

        if sparse:
            if (parent.x + 1) % mem_column == 0:
                comp.sb_delay_rv.append(
                    comp.delays[
                        f"MEM_B{parent.bit_width}_valid_{side_to_dir[parent.side]}_{side_to_dir[next_sb.side]}"
                    ]
                )
            else:
                comp.sb_delay_rv.append(
                    comp.delays[
                        f"PE_B{parent.bit_width}_valid_{side_to_dir[parent.side]}_{side_to_dir[next_sb.side]}"
                    ]
                )

            if (parent.x + 1) % mem_column == 0:
                comp.sb_delay_rv.append(
                    comp.delays[
                        f"MEM_B{parent.bit_width}_ready_{side_to_dir[next_sb.side]}_{side_to_dir[parent.side]}"
                    ]
                )
            else:
                comp.sb_delay_rv.append(
                    comp.delays[
                        f"PE_B{parent.bit_width}_ready_{side_to_dir[next_sb.side]}_{side_to_dir[parent.side]}"
                    ]
                )


def calc_fifo_to_out(graph, node, parent, comp, mem_tile_column):
    assert graph.sparse

    if not (
        isinstance(graph.sources[parent][0], RouteNode)
        and graph.sources[parent][0].route_type == RouteType.SB
    ):
        if (node.x + 1) % mem_tile_column == 0:
            tile_suffix = "mem"
        else:
            tile_suffix = "pe"

        if isinstance(parent, RouteNode) and parent.route_type == RouteType.REG:
            # Split fifo to SB out
            prefix = "split"
        else:
            # Sparse prim fifo to SB out
            prefix = "prim"

        comp.sb_delay.append(comp.delays[f"{prefix}_fifo_to_sb_out_{tile_suffix}"])

        comp.sb_delay_rv.append(
            comp.delays[f"{prefix}_fifo_to_sb_out_{tile_suffix}_ready"]
        )

        # Ready and-ing logic to produce valid
        comp.sb_delay_rv.append(comp.delays[f"ready_and_valid_{tile_suffix}"])


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
                if node.input_port_break_path[parent.port]:
                    comp = PathComponents()
            else:
                if node.route_type == RouteType.PORT and isinstance(parent, TileNode):
                    if parent.tile_type == TileType.PE:
                        comp.pes += 1
                    elif parent.tile_type == TileType.MEM:
                        comp.mems += 1
                    elif (
                        parent.tile_type == TileType.IO16
                        or parent.tile_type == TileType.IO1
                    ):
                        comp.glbs += 1

                elif node.route_type == RouteType.SB:
                    calc_sb_delay(
                        graph, node, parent, comp, mem_tile_column, graph.sparse
                    )

                elif node.route_type == RouteType.RMUX:
                    if graph.sparse:
                        calc_fifo_to_out(graph, node, parent, comp, mem_tile_column)
                    else:
                        # Make sure to check this later, not sure if we only want to count rmux when reg is used
                        if (
                            isinstance(parent, RouteNode)
                            and parent.route_type == RouteType.REG
                        ):
                            comp.rmux += 1
                        if parent.route_type != RouteType.REG:
                            comp.available_regs += 1
                elif (
                    node.route_type == RouteType.PORT
                    and isinstance(parent, TileNode)
                    and parent.tile_type == TileType.IO16
                ):
                    comp.sb_delay_rv.append(300)
                    comp.sb_delay_rv.append(635)

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

    clock_speed, crit_path, crit_nodes = sta(routing_result_graph)

    return clock_speed


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
