import sys
import copy
import os
import argparse
import re
import itertools
import glob
import json
from typing import Dict, List, NoReturn, Tuple, Set
from archipelago.pnr_graph import (
    KernelNodeType,
    RoutingResultGraph,
    construct_graph,
    construct_kernel_graph,
    TileType,
    RouteType,
    TileNode,
    RouteNode,
)
from archipelago.sta import sta, load_graph


def verboseprint(*args, **kwargs):
    print(*args, **kwargs)


# verboseprint = lambda *a, **k: None


class bcolors:
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


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

    if graph.sparse:
        break_idx += 3
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


def break_at(graph, node1, id_to_name, placement, routing):
    path = []
    curr_node = node1
    kernel = curr_node.kernel

    while len(graph.sinks[curr_node]) == 1:
        if (
            len(graph.sources[graph.sinks[curr_node][0]]) > 1
            or graph.sinks[curr_node][0].kernel != kernel
        ):
            break
        curr_node = graph.sinks[curr_node][0]

    idx = 0
    while len(graph.sources[curr_node]) == 1:
        if (
            len(graph.sinks[curr_node]) > 1
            or graph.sources[curr_node][0].kernel != kernel
        ):
            break
        path.append((curr_node, idx))
        curr_node = graph.sources[curr_node][0]
        if curr_node in graph.get_ponds():
            break

    if curr_node in graph.get_ponds():
        verboseprint("\t\tFound pond for branch delay matching", curr_node)
        curr_node.input_port_latencies["data_in_pond"] += 1
        return

    if len(path) == 0:
        raise ValueError(f"Cant break at node: {node1}")
    path.reverse()
    ret = []
    for p in path:
        ret.append((p[0], idx))
        idx += 1
    break_crit_path(graph, id_to_name, ret, placement, routing)


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


# def break_at_mems(graph, id_to_name, placement, routes, sparse):
#     if sparse:
#         return
#     for mem in graph.get_mems():
#         for port in graph.sinks[mem]:
#             for sb_port in graph.sinks[port]:
#                 rmux = graph.sinks[sb_port][0]
#                 assert isinstance(rmux, RouteNode)
#                 assert rmux.route_type == RouteType.RMUX, str(rmux)
#                 crit_path = [(sb_port, 0), (rmux, 1)]
#                 break_crit_path(graph, id_to_name, crit_path, placement, routes)
#                 reg = graph.sinks[graph.sinks[sb_port][0]][0]
#                 reg.input_port_latencies["reg"] = 0
#                 reg.input_port_break_path["reg"] = True


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

        if len(graph.sources[node]) == 0:
            if node in graph.get_pes():
                cycles = {None}
            else:
                cycles = {0}

        for parent in graph.sources[node]:
            if parent not in node_cycles:
                c = 0
            else:
                c = node_cycles[parent]

            if c != None and len(graph.sinks[node]) > 0 and isinstance(node, TileNode):
                c += node.input_port_latencies[parent.port]

            # Flush signals shouldn't be considered here
            if "reset" not in node.kernel:
                cycles.add(c)

        if None in cycles:
            cycles.remove(None)

        if len(graph.sources[node]) > 1 and len(cycles) > 1:
            verboseprint(f"Incorrect node delay: {node} {cycles}")

        if len(cycles) > 0:
            node_cycles[node] = max(cycles)
        else:
            node_cycles[node] = None


def branch_delay_match_within_kernels(graph, id_to_name, placement, routing):
    nodes = graph.topological_sort()
    node_cycles = {}

    for node in nodes:
        cycles = set()

        if node.kernel not in node_cycles:
            node_cycles[node.kernel] = {}

        if len(graph.sources[node]) == 0:
            if node in graph.get_pes():
                cycles = {None}
            elif node in graph.get_input_ios():
                cycles = {node.input_port_latencies["output"]}
            else:
                cycles = {0}

        for parent in graph.sources[node]:
            if parent not in node_cycles[node.kernel]:
                c = 0
            else:
                c = node_cycles[node.kernel][parent]

            if c != None and isinstance(node, TileNode):
                c += node.input_port_latencies[parent.port]

            cycles.add(c)

        if None in cycles:
            cycles.remove(None)

        if len(cycles) > 1:
            verboseprint(
                f"\tIncorrect delay within kernel: {node.kernel} {node} {cycles}"
            )
            verboseprint(f"\tFixing branching delays at: {node} {cycles}")
            source_cycles = [
                node_cycles[node.kernel][source]
                for source in graph.sources[node]
                if node_cycles[node.kernel][source] != None
            ]
            max_parent_cycles = max(source_cycles)
            for parent in graph.sources[node]:
                if node_cycles[node.kernel][parent] != max_parent_cycles:
                    for _ in range(
                        max_parent_cycles - node_cycles[node.kernel][parent]
                    ):
                        break_at(
                            graph,
                            graph.sources[parent][0],
                            id_to_name,
                            placement,
                            routing,
                        )

        if len(cycles) > 0:
            node_cycles[node.kernel][node] = max(cycles)
        else:
            node_cycles[node.kernel][node] = None

    for kernel, node_delays in node_cycles.items():
        kernel_output_delays = set()
        kernel_output_nodes_and_delays = []
        for node in graph.get_outputs_of_kernel(kernel):
            if node_delays[node] != None:
                kernel_output_delays.add(node_delays[node])
                kernel_output_nodes_and_delays.append((node, node_delays[node]))

        if len(kernel_output_delays) > 1:
            verboseprint(
                f"\tIncorrect delay at output of kernel: {kernel} {kernel_output_delays}"
            )
            max_parent_cycles = max(kernel_output_delays)
            for node, delay in kernel_output_nodes_and_delays:
                if delay != max_parent_cycles:
                    verboseprint(
                        f"\tFixing branching delays at: {node} {max_parent_cycles - delay}"
                    )
                    for _ in range(max_parent_cycles - delay):
                        break_at(graph, node, id_to_name, placement, routing)

    kernel_latencies = {}
    for kernel in node_cycles:
        kernel_cycles = [cyc for cyc in node_cycles[kernel].values() if cyc != None]
        kernel_cycles.append(0)
        kernel_latencies[kernel] = max(kernel_cycles)

    return kernel_latencies


def branch_delay_match_kernels(kernel_graph, graph, id_to_name, placement, routing):
    nodes = kernel_graph.topological_sort()
    node_cycles = {}

    for node in nodes:
        cycles = set()

        if len(kernel_graph.sources[node]) == 0:
            if (
                node.kernel_type == KernelNodeType.COMPUTE
                or node.kernel_type == KernelNodeType.MEM
            ):
                cycles = {None}
            else:
                cycles = {0}

        for parent in kernel_graph.sources[node]:
            if parent not in node_cycles:
                c = 0
            else:
                c = node_cycles[parent]

            if c is not None:
                c += node.latency

            if not (
                "reset" in parent.kernel
                or (parent.kernel_type == KernelNodeType.MEM and str(parent)[0] == "m")
            ):
                cycles.add(c)

        if None in cycles:
            cycles.remove(None)

        if len(kernel_graph.sources[node]) > 1 and len(cycles) > 1:
            verboseprint(f"\tIncorrect kernel delay: {node} {cycles}")

            source_cycles = [
                node_cycles[source]
                for source in kernel_graph.sources[node]
                if node_cycles[source] != None
            ]
            max_cycle = max(source_cycles)
            for source in kernel_graph.sources[node]:
                if node_cycles[source] != None and node_cycles[source] != max_cycle:
                    verboseprint(
                        f"\tFixing kernel delays at: {source} {max_cycle - node_cycles[source]}"
                    )
                    add_delay_to_kernel(
                        graph,
                        source.kernel,
                        max_cycle - node_cycles[source],
                        id_to_name,
                        placement,
                        routing,
                    )
        if len(cycles) > 0:
            node_cycles[node] = max(cycles)
        else:
            node_cycles[node] = None


def flush_cycles(
    graph, id_to_name, harden_flush, pipeline_config_interval, pes_with_packed_ponds
):
    if harden_flush:
        flush_cycles = {}
        for mem in graph.get_mems() + graph.get_ponds():
            if mem.y == 0 or pipeline_config_interval == 0:
                flush_cycles[mem] = 0
            else:
                flush_cycles[mem] = (mem.y - 1) // pipeline_config_interval

        for pe in graph.get_pes():
            if (
                pes_with_packed_ponds is not None
                and pe.tile_id in pes_with_packed_ponds
            ):
                pond = pes_with_packed_ponds[pe.tile_id]
                if pe.y == 0 or pipeline_config_interval == 0:
                    flush_cycles[pond] = 0
                else:
                    flush_cycles[pond] = (pe.y - 1) // pipeline_config_interval

    else:
        for io in graph.get_input_ios():
            if io.kernel == "io1in_reset":
                break
        assert io.kernel == "io1in_reset"
        flush_cycles = {}

        for mem in graph.get_mems() + graph.get_ponds():
            for parent_node in graph.sources[mem]:
                if parent_node.port == "flush":
                    break
            if parent_node.port != "flush":
                continue

            curr_node = mem
            flush_cycles[mem] = 0
            while parent_node != io:
                if isinstance(curr_node, TileNode):
                    flush_cycles[mem] += curr_node.input_port_latencies[
                        parent_node.port
                    ]
                curr_node = parent_node
                parent_node = graph.sources[parent_node][0]

    max_flush_cycles = max(flush_cycles.values())
    for mem, flush_c in flush_cycles.items():
        flush_cycles[mem] = max_flush_cycles - flush_c

    return flush_cycles, max_flush_cycles


def find_closest_match(kernel_target, candidates):
    junk = ["hcompute", "cgra", "global", "wrapper", "clkwrk", "stencil", "op"]

    cleaned_candidates = candidates.copy()
    for idx, key in enumerate(candidates):
        cleaned_candidates[idx] = cleaned_candidates[idx].split("_")
        for j in junk:
            if j in cleaned_candidates[idx]:
                cleaned_candidates[idx] = [c for c in cleaned_candidates[idx] if c != j]
    kernel_target = kernel_target.split("_")
    if "clkwrk" in kernel_target:
        del kernel_target[
            kernel_target.index("clkwrk") : kernel_target.index("clkwrk") + 2
        ]

    for j in junk:
        if j in kernel_target:
            kernel_target = [k for k in kernel_target if k != j]

    matches_and_ratios = []

    for idx, candidate in enumerate(cleaned_candidates):
        ratio = 0
        if "glb" not in candidate:
            for a in candidate:
                if a in kernel_target:
                    ratio += 1
        matches_and_ratios.append((candidates[idx], ratio))

    return max(matches_and_ratios, key=lambda item: item[1])[0]


def calculate_latencies(kernel_graph, kernel_latencies):
    nodes = kernel_graph.topological_sort()
    new_latencies = {}
    flush_latencies = {}

    for node in kernel_graph.nodes:
        if node.kernel_type == KernelNodeType.COMPUTE:
            new_latencies[node.kernel] = node.latency

    for node16 in new_latencies:
        for node1 in new_latencies:
            if (
                node16 != node1
                and node16.split("_write")[0].replace("io16", "io1")
                == node1.split("_write")[0]
            ):
                new_latencies[node16] -= new_latencies[node1]
                new_latencies[node1] = 0

    # Unfortunately exact matches between kernels and memories dont exist, so we have to look them up
    sorted_new_latencies = {}
    for k in sorted(new_latencies, key=lambda a: len(str(a)), reverse=True):
        sorted_new_latencies[k] = new_latencies[k]

    for graph_kernel, lat in sorted_new_latencies.items():
        if "op_" in graph_kernel and graph_kernel.split("op_")[1] in kernel_latencies:
            kernel_latencies[graph_kernel.split("op_")[1]] = lat
        elif "io16" in graph_kernel:
            # Used for input/output kernels
            match = find_closest_match(graph_kernel, list(kernel_latencies.keys()))
            kernel_latencies[match] = lat
    return kernel_latencies


def update_kernel_latencies(
    dir_name,
    graph,
    id_to_name,
    placement,
    routing,
    harden_flush,
    pipeline_config_interval,
    pes_with_packed_ponds,
    sparse,
):
    if sparse:
        return

    kernel_latencies = branch_delay_match_within_kernels(
        graph, id_to_name, placement, routing
    )

    kernel_graph = construct_kernel_graph(graph, kernel_latencies)

    branch_delay_match_kernels(kernel_graph, graph, id_to_name, placement, routing)

    branch_delay_match_all_nodes(graph, id_to_name, placement, routing)

    flush_latencies, max_flush_cycles = flush_cycles(
        graph, id_to_name, harden_flush, pipeline_config_interval, pes_with_packed_ponds
    )

    for node in kernel_graph.nodes:
        if "io16in" in node.kernel or "io1in" in node.kernel:
            node.latency -= max_flush_cycles
            assert (
                node.latency >= 0
            ), f"{node.kernel} has negative compute kernel latency"

    kernel_latencies_file = glob.glob(f"{dir_name}/*_compute_kernel_latencies.json")[0]
    flush_latencies_file = kernel_latencies_file.replace(
        "compute_kernel_latencies", "flush_latencies"
    )
    pond_latencies_file = kernel_latencies_file.replace(
        "compute_kernel_latencies", "pond_latencies"
    )

    assert os.path.exists(kernel_latencies_file)

    f = open(kernel_latencies_file, "r")
    existing_kernel_latencies = json.load(f)

    matched_kernel_latencies = calculate_latencies(
        kernel_graph, existing_kernel_latencies
    )
    matched_flush_latencies = {
        id_to_name[str(mem_id)]: latency for mem_id, latency in flush_latencies.items()
    }

    pond_latencies = {}
    for pond_node in graph.get_ponds():
        for port, lat in pond_node.input_port_latencies.items():
            if port != "flush":
                pond_latencies[id_to_name[pond_node.tile_id]] = lat

    fout = open(kernel_latencies_file, "w")
    fout.write(json.dumps(matched_kernel_latencies, indent=4))

    fout = open(flush_latencies_file, "w")
    fout.write(json.dumps(matched_flush_latencies, indent=4))

    fout = open(pond_latencies_file, "w")
    fout.write(json.dumps(pond_latencies, indent=4))


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


def pipeline_pnr(
    app_dir,
    placement,
    routing,
    id_to_name,
    netlist,
    load_only,
    harden_flush,
    pipeline_config_interval,
    pes_with_packed_ponds,
    sparse,
):

    if load_only:
        id_to_name_filename = os.path.join(app_dir, f"design.id_to_name")
        if os.path.isfile(id_to_name_filename):
            id_to_name = load_id_to_name(id_to_name_filename)
        return placement, routing, id_to_name

    placement_save = copy.deepcopy(placement)
    routing_save = copy.deepcopy(routing)
    id_to_name_save = copy.deepcopy(id_to_name)

    if "PIPELINED" in os.environ and os.environ["PIPELINED"].isnumeric():
        pe_cycles = int(os.environ["PIPELINED"])
    else:
        pe_cycles = 1

    if "IO_DELAY" in os.environ and os.environ["IO_DELAY"] == "0":
        io_cycles = 0
    else:
        io_cycles = 1

    graph = construct_graph(
        placement,
        routing,
        id_to_name,
        netlist,
        pe_latency=pe_cycles,
        pond_latency=0,
        io_latency=io_cycles,
        sparse=sparse,
    )

    break_at_mems(graph, id_to_name, placement, routing, sparse)

    print("\nApplication Frequency:")
    curr_freq, crit_path, crit_nets = sta(graph)

    update_kernel_latencies(
        app_dir,
        graph,
        id_to_name,
        placement,
        routing,
        harden_flush,
        pipeline_config_interval,
        pes_with_packed_ponds,
        sparse,
    )

    if "POST_PNR_ITR" in os.environ:
        if os.environ["POST_PNR_ITR"] == "max":
            max_itr = None
        else:
            max_itr = int(os.environ["POST_PNR_ITR"])

        curr_freq = 0
        itr = 0

        while max_itr == None:
            try:
                break_crit_path(graph, id_to_name, crit_path, placement, routing)
                graph.regs = None
                update_kernel_latencies(
                    app_dir,
                    graph,
                    id_to_name,
                    placement,
                    routing,
                    harden_flush,
                    pipeline_config_interval,
                    pes_with_packed_ponds,
                    sparse,
                )

                print("\nIteration", itr + 1, "frequency")
                curr_freq, crit_path, crit_nets = sta(graph)
            except:
                max_itr = itr
            itr += 1

        print("\nCan break", max_itr, "critical paths")

        # Reloading best result
        id_to_name = id_to_name_save
        placement = placement_save
        routing = routing_save
        graph = construct_graph(
            placement,
            routing,
            id_to_name,
            netlist,
            pe_latency=pe_cycles,
            pond_latency=0,
            io_latency=io_cycles,
            sparse=sparse,
        )

        break_at_mems(graph, id_to_name, placement, routing, sparse)

        starting_regs = graph.added_regs

        update_kernel_latencies(
            app_dir,
            graph,
            id_to_name,
            placement,
            routing,
            harden_flush,
            pipeline_config_interval,
            pes_with_packed_ponds,
            sparse,
        )

        for _ in range(max_itr):
            curr_freq, crit_path, crit_nets = sta(graph)
            break_crit_path(graph, id_to_name, crit_path, placement, routing)

        update_kernel_latencies(
            app_dir,
            graph,
            id_to_name,
            placement,
            routing,
            harden_flush,
            pipeline_config_interval,
            pes_with_packed_ponds,
            sparse,
        )
        print("\nFinal application frequency:")
        curr_freq, crit_path, crit_nets = sta(graph)

        if max_itr == 0:
            print(bcolors.WARNING + "\nCouldn't break any paths" + bcolors.ENDC)
        else:
            print(bcolors.OKGREEN + "\nBroke", max_itr, "critical paths" + bcolors.ENDC)

        print(
            "\nAdded", graph.added_regs - starting_regs, "registers to routing graph\n"
        )

    freq_file = os.path.join(app_dir, "design.freq")
    fout = open(freq_file, "w")
    fout.write(f"{curr_freq}\n")

    dump_routing_result(app_dir, routing)
    dump_placement_result(app_dir, placement, id_to_name)
    dump_id_to_name(app_dir, id_to_name)

    return placement, routing, id_to_name
