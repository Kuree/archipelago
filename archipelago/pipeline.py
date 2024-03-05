import copy
import os
import glob
import json
from archipelago.pnr_graph import (
    KernelNodeType,
    construct_graph,
    construct_kernel_graph,
    TileType,
    RouteType,
    TileNode,
    RouteNode,
)
from archipelago.sta import sta
import pythunder


# def verboseprint(*args, **kwargs):
#     print(*args, **kwargs)


verboseprint = lambda *a, **k: None


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
    verboseprint("\nBreaking net:", net_id, "Kernel:", kernel)

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
        verboseprint("\nBreaking net:", net_id, "Kernel:", kernel)

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


def exhaustive_pipe(graph, id_to_name, placement, routing):
    for node in graph.nodes:
        if node in graph.get_tiles() or len(graph.sinks[node]) > 1:
            for sink in graph.sinks[node]:
                path = []
                curr_node = sink
                while True:
                    path.append((curr_node, len(path)))
                    if len(graph.sinks[curr_node]) != 1:
                        break
                    curr_node = graph.sinks[curr_node][0]

                for idx in range(len(path)):
                    if graph.sparse:
                        if idx + 4 >= len(path):
                            break
                        if (
                            isinstance(path[idx][0], RouteNode)
                            and path[idx][0].route_type == RouteType.SB
                            and path[idx + 1][0].route_type == RouteType.RMUX
                            and isinstance(path[idx + 2][0], RouteNode)
                            and path[idx + 2][0].route_type == RouteType.SB
                            and isinstance(path[idx + 3][0], RouteNode)
                            and path[idx + 3][0].route_type == RouteType.SB
                            and isinstance(path[idx + 4][0], RouteNode)
                            and path[idx + 4][0].route_type == RouteType.RMUX
                        ):
                            try:
                                break_crit_path(
                                    graph,
                                    id_to_name,
                                    path[idx : idx + 5],
                                    placement,
                                    routing,
                                )
                            except:
                                verboseprint("Skip")
                    else:
                        if idx + 1 >= len(path):
                            break
                        if (
                            isinstance(path[idx][0], RouteNode)
                            and path[idx][0].route_type == RouteType.SB
                            and isinstance(path[idx + 1][0], RouteNode)
                            and path[idx + 1][0].route_type == RouteType.RMUX
                        ):
                            try:
                                break_crit_path(
                                    graph,
                                    id_to_name,
                                    path[idx : idx + 2],
                                    placement,
                                    routing,
                                )
                            except:
                                verboseprint("Skip")


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


def find_closest_match(kernel_target, candidates):
    candidates = [c for c in candidates if "io1_" not in c]

    if "op_" + kernel_target in candidates:
        return "op_" + kernel_target

    kernel_target_out = kernel_target + "_write"
    for c in candidates:
        if kernel_target_out in c:
            return c

    kernel_target_in = kernel_target + "_read"
    kernel_target_in = kernel_target_in.replace(
        "global_wrapper_global_wrapper", "global_wrapper_glb"
    )
    for c in candidates:
        if kernel_target_in in c:
            return c

    kernel_target_in = kernel_target + "_read"
    kernel_target_in = kernel_target_in.replace("cgra", "glb")
    for c in candidates:
        if kernel_target_in in c:
            return c

    print("No match for", kernel_target)


def branch_delay_match_within_kernels(
    graph, id_to_name, placement, routing, kernel_latencies, port_remap
):
    port_remap_r = {v: k for k, v in port_remap["pe"].items()}
    port_remap_r["reg"] = "reg"
    nodes = graph.topological_sort()
    nodes.reverse()
    node_cycles = {}

    for node in nodes:
        if node.kernel not in node_cycles:
            node_cycles[node.kernel] = {}

        cycles = set()
        if len(graph.sinks[node]) == 0:
            cycles = {0}

        for sink in graph.sinks[node]:
            if sink not in node_cycles[node.kernel]:
                node_cycles[node.kernel][sink] = 0

            c = node_cycles[node.kernel][sink]

            if c != None and isinstance(sink, TileNode):
                c += sink.input_port_latencies[node.port]
            elif node in graph.get_input_ios():
                # Need special case for input IOs
                c += node.input_port_latencies["output"]

            if (
                isinstance(node, TileNode)
                and node.tile_type == TileType.PE
                and sink.port == "PondTop_output_width_17_num_0"
            ):
                continue
            cycles.add(c)

        if None in cycles:
            cycles.remove(None)

        if len(cycles) > 1:
            if "IO2MEM_REG_CHAIN" in os.environ or "MEM2PE_REG_CHAIN" in os.environ:
                continue
            verboseprint(
                f"\tIncorrect delay within kernel: {node.kernel} {node} {cycles}"
            )
            verboseprint(f"\tFixing branching delays at: {node} {cycles}")
            sink_cycles = [
                node_cycles[node.kernel][sink]
                for sink in graph.sinks[node]
                if node_cycles[node.kernel][sink] != None
            ]
            max_sink_cycles = max(sink_cycles)
            for sink in graph.sinks[node]:
                for _ in range(max_sink_cycles - node_cycles[node.kernel][sink]):
                    break_at(
                        graph,
                        node,
                        id_to_name,
                        placement,
                        routing,
                    )
            node_cycles[node.kernel][node] = max(cycles)
        elif len(cycles) == 1:
            node_cycles[node.kernel][node] = max(cycles)
        else:
            node_cycles[node.kernel][node] = None

    # Only certain inputs of compute kernels can have different latencies (dictated by clockwork and H2H)
    # First determine which nodes can have unique latencies
    ports_with_unique_latenices = {}
    for kernel, latency_dict in kernel_latencies.items():
        if "_glb_" in kernel:
            continue
        match = find_closest_match(kernel, list(node_cycles.keys()))
        if match is not None:
            ports_with_unique_latenices[match] = []
            for kernel_port, d1 in latency_dict.items():
                if d1["pe_port"] != []:
                    port_nodes = []
                    for compute_file_tile, compute_file_port in d1["pe_port"]:
                        found = False
                        for pe in graph.get_tiles():
                            if (
                                graph.id_to_name[str(pe)]
                                == f"{match}$inner_compute${compute_file_tile}"
                            ):
                                found_port = False
                                for source in graph.sources[pe]:
                                    if source.port in port_remap_r:
                                        port = port_remap_r[source.port]
                                        if port == compute_file_port:
                                            found = True
                                            found_port = True
                                            port_nodes.append(source)

                                for source_node, dest_node in graph.removed_edges:
                                    if dest_node == pe:
                                        if source_node.port in port_remap_r:
                                            port = port_remap_r[source_node.port]
                                            if port == compute_file_port:
                                                found = True
                                                found_port = True
                                                port_nodes.append(source_node)

                                if not found_port:
                                    print("Couldn't find pe port")
                                    print(latency_dict)
                                    breakpoint()

                        if not found:
                            print("Couldn't find pe")
                            print(latency_dict)
                            breakpoint()

                    ports_with_unique_latenices[match].append(port_nodes)

    # Then branch delay match the nodes without unique latencies
    for kernel in node_cycles:
        if kernel not in ports_with_unique_latenices:
            continue

        for nodes_with_same_latency in ports_with_unique_latenices[kernel]:
            kernel_input_latencies = [
                node_cycles[kernel][kernel_input]
                for kernel_input in nodes_with_same_latency
            ]

            for node_with_same_latency in nodes_with_same_latency:
                same_latency = max(kernel_input_latencies)

                if (
                    node_cycles[kernel][node_with_same_latency] != same_latency
                    and node_with_same_latency
                    not in ports_with_unique_latenices[kernel]
                ):
                    verboseprint(
                        f"\tIncorrect delay between ports of kernel: {kernel} {node_with_same_latency} {node_cycles[kernel][node_with_same_latency]} {same_latency}"
                    )
                    verboseprint(
                        f"\tFixing branching delays at: {node_with_same_latency}"
                    )
                    for sink in graph.sinks[node_with_same_latency]:
                        for _ in range(
                            same_latency - node_cycles[kernel][node_with_same_latency]
                        ):
                            break_at(
                                graph,
                                sink,
                                id_to_name,
                                placement,
                                routing,
                            )
                    node_cycles[node.kernel][node_with_same_latency] = same_latency

    kernel_latencies = {}
    for kernel in node_cycles:
        kernel_latencies[kernel] = max(node_cycles[kernel].values())

    return kernel_latencies, node_cycles


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


def find_stencil_valid_mem(graph, kernel):
    for node in graph.nodes:
        if node.kernel == kernel:
            break

    curr_node = node

    while True:
        if isinstance(curr_node, TileNode) and curr_node.tile_type == TileType.MEM:
            return curr_node

        if len(graph.sources[curr_node]) == 0:
            return None

        curr_node = graph.sources[curr_node][0]


def calculate_latencies(
    graph, kernel_graph, node_latencies, kernel_latencies, port_remap, instance_to_instr
):
    port_remap_r = {v: k for k, v in port_remap["pe"].items()}

    max_latencies = {}

    for node in kernel_graph.nodes:
        if node.kernel_type == KernelNodeType.COMPUTE:
            max_latencies[node.kernel] = node.latency

    stencil_valid_adjust = {}

    for node16 in max_latencies:
        for node1 in max_latencies:
            if (
                node16 != node1
                and node16.split("_write")[0].replace("io16", "io1")
                == node1.split("_write")[0]
            ):
                max_diff = -(max_latencies[node16] + 2)

                # Need to absorb the added latency of the stencil valids into either the compute kernel or the stencil valid schedule generator
                max_latencies[node16] -= max_latencies[node1]
                max_latencies[node1] = 0

                if max_latencies[node16] < max_diff:
                    raise Exception(
                        f"Can't absorb stencil valid latency of {max_latencies[node16]} into compute kernel"
                    )

                # if max_latencies[node16] < 0:
                stencil_valid_mem = find_stencil_valid_mem(graph, node1)
                # need to adjust stencil valid latency
                stencil_valid_adjust[stencil_valid_mem.kernel] = max_latencies[node16]
                max_latencies[node16] = 0

    # Alright I know this next part is gross but I think it works?
    # Kernel latencies are from the file passed to clockwork, need to match the latencies in the routing graph to this
    for kernel, latency_dict in kernel_latencies.items():
        # glb kernels are not in the routing graph
        if "_glb_" in kernel:
            continue

        # Find the closest matched kernel in the routing graph, we don't have exact matches because renaming
        match = find_closest_match(kernel, list(node_latencies.keys()))
        if match is not None:
            for kernel_port, d1 in latency_dict.items():
                if d1["pe_port"] == [] and match in max_latencies:
                    kernel_latencies[kernel][kernel_port][
                        "latency"
                    ] = max_latencies[match]
                elif d1["pe_port"] != []:
                    found = False
                    for compute_file_tile, compute_file_port in d1["pe_port"]:
                        # Within this loop, all the ports should have the same latency
                        found_lat = None
                        for pe in graph.get_tiles():
                            if (
                                graph.id_to_name[str(pe)]
                                == f"{match}$inner_compute${compute_file_tile}"
                            ):
                                found_port = False
                                for source in graph.sources[pe]:
                                    if source.port in port_remap_r:
                                        port = port_remap_r[source.port]
                                        if port == compute_file_port:
                                            reg = graph.get_connected_reg(source)
                                            if reg is not None:
                                                lat = node_latencies[match][reg]
                                            else:
                                                lat = node_latencies[match][source]

                                            if found_lat is not None:
                                                assert (
                                                    lat == found_lat
                                                ), f"Found multiple latencies for {kernel} {kernel_port} {compute_file_tile} {compute_file_port} {lat} {found_lat}"
                                            kernel_latencies[kernel][kernel_port]["latency"] = lat
                                            found = True
                                            found_port = True
                                            found_lat = lat
                                            break
                                if not found_port:
                                    found = True
                                    kernel_latencies[kernel][kernel_port][
                                        "latency"
                                    ] = node_latencies[match][graph.sources[pe][0]]

                    if not found:
                        print("Couldn't find tile port in kernel latencies", kernel)

    return kernel_latencies, stencil_valid_adjust


def update_kernel_latencies(
    dir_name,
    graph,
    id_to_name,
    placement,
    routing,
    existing_kernel_latencies,
    harden_flush,
    instance_to_instr,
    pipeline_config_interval,
    pes_with_packed_ponds,
    sparse,
):
    if sparse:
        return

    port_remap = json.load(open(f"{dir_name}/design.port_remap"))

    kernel_latencies, node_latencies = branch_delay_match_within_kernels(
        graph, id_to_name, placement, routing, existing_kernel_latencies, port_remap
    )

    kernel_graph = construct_kernel_graph(graph, kernel_latencies)

    branch_delay_match_kernels(kernel_graph, graph, id_to_name, placement, routing)

    # branch_delay_match_all_nodes(graph, id_to_name, placement, routing)

    flush_latencies, max_flush_cycles = flush_cycles(
        graph, id_to_name, harden_flush, pipeline_config_interval, pes_with_packed_ponds
    )
    for node in kernel_graph.nodes:
        if "io16in" in node.kernel or "io1in" in node.kernel:
            node.latency -= max_flush_cycles
            assert (
                node.latency >= 0
            ), f"{node.kernel} has negative compute kernel latency"

    matched_kernel_latencies, stencil_valid_adjust = calculate_latencies(
        graph,
        kernel_graph,
        node_latencies,
        existing_kernel_latencies,
        port_remap,
        instance_to_instr,
    )
    # updated_kernel_latencies.json only for residual add and manual placed resnet for now
    if os.path.exists(f"{dir_name}/updated_kernel_latencies.json"):
        updated_kernel_latencies = json.load(open(f"{dir_name}/updated_kernel_latencies.json"))
        for kernel, latency_dict in matched_kernel_latencies.items():
            if "hcompute_output_cgra_stencil" in kernel:
                for kernel_port, d1 in latency_dict.items():
                    if "input_cgra_stencil" or "in2_output_cgra_stencil" in kernel_port:
                        d1["latency"] = updated_kernel_latencies[kernel][kernel_port]["latency"]
            if "_glb_" in kernel:
                matched_kernel_latencies[kernel] = updated_kernel_latencies[kernel]
    # ub_latency.json only for manual placed resnet
    if os.path.exists(f"{dir_name}/ub_latency.json"):
        ub_latencies = json.load(open(f"{dir_name}/ub_latency.json"))
        for kernel, latency_dict in matched_kernel_latencies.items():
            if "hcompute_input_cgra_stencil" in kernel:
                for kernel_port, d1 in latency_dict.items():
                    port_num = kernel_port.split("_")[-1]
                    d1["latency"] = ub_latencies["input_cgra_stencil"][port_num]["latency"]
            if "hcompute_kernel_cgra_stencil" in kernel:
                for kernel_port, d1 in latency_dict.items():
                    d1["latency"] = min(value["latency"] for value in ub_latencies["kernel_cgra_stencil"].values())
    matched_flush_latencies = {
        id_to_name[str(mem_id)]: latency for mem_id, latency in flush_latencies.items()
    }

    pond_latencies = {}
    for pond_node in graph.get_ponds():
        for port, lat in pond_node.input_port_latencies.items():
            if port != "flush":
                pond_latencies[id_to_name[pond_node.tile_id]] = lat

    kernel_latencies_file = glob.glob(f"{dir_name}/*_compute_kernel_latencies.json")[0]

    flush_latencies_file = kernel_latencies_file.replace(
        "compute_kernel_latencies", "flush_latencies"
    )
    pond_latencies_file = kernel_latencies_file.replace(
        "compute_kernel_latencies", "pond_latencies"
    )
    stencil_valid_latencies_file = kernel_latencies_file.replace(
        "compute_kernel_latencies", "stencil_valid_latencies"
    )

    fout = open(kernel_latencies_file, "w")
    fout.write(json.dumps(matched_kernel_latencies, indent=4))

    fout = open(flush_latencies_file, "w")
    fout.write(json.dumps(matched_flush_latencies, indent=4))

    fout = open(pond_latencies_file, "w")
    fout.write(json.dumps(pond_latencies, indent=4))

    fout = open(stencil_valid_latencies_file, "w")
    fout.write(json.dumps(stencil_valid_adjust, indent=4))


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
    instance_to_instr,
    pipeline_config_interval,
    pes_with_packed_ponds,
    sparse,
):
    if load_only:
        packed_file = os.path.join(app_dir, "design.packed")
        id_to_name = pythunder.io.load_id_to_name(packed_file)
        return placement, routing, id_to_name

    placement_save = copy.deepcopy(placement)
    routing_save = copy.deepcopy(routing)
    id_to_name_save = copy.deepcopy(id_to_name)

    existing_kernel_latencies = {}
    if not sparse:
        kernel_latencies_file = glob.glob(f"{app_dir}/*_compute_kernel_latencies.json")[0]
        existing_kernel_latencies = json.load(open(kernel_latencies_file, "r"))

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
        existing_kernel_latencies,
        pe_latency=pe_cycles,
        pond_latency=0,
        io_latency=io_cycles,
        sparse=sparse,
    )

    print("\nApplication Frequency:")
    curr_freq, crit_path, crit_nets = sta(graph)

    update_kernel_latencies(
        app_dir,
        graph,
        id_to_name,
        placement,
        routing,
        existing_kernel_latencies,
        harden_flush,
        instance_to_instr,
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
                    existing_kernel_latencies,
                    harden_flush,
                    instance_to_instr,
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
            existing_kernel_latencies,
            pe_latency=pe_cycles,
            pond_latency=0,
            io_latency=io_cycles,
            sparse=sparse,
        )
        starting_regs = graph.added_regs

        update_kernel_latencies(
            app_dir,
            graph,
            id_to_name,
            placement,
            routing,
            existing_kernel_latencies,
            harden_flush,
            instance_to_instr,
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
            existing_kernel_latencies,
            harden_flush,
            instance_to_instr,
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
    elif "EXHAUSTIVE_PIPE" in os.environ:
        starting_regs = graph.added_regs
        exhaustive_pipe(graph, id_to_name, placement, routing)
        curr_freq, crit_path, crit_nets = sta(graph)
        print(
            "\nAdded", graph.added_regs - starting_regs, "registers to routing graph\n"
        )

    freq_file = os.path.join(app_dir, "design.freq")
    fout = open(freq_file, "w")
    fout.write(f"{curr_freq}\n")

    dump_routing_result(app_dir, routing)
    dump_placement_result(app_dir, placement, id_to_name)

    return placement, routing, id_to_name
