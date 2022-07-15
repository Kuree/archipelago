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

            if c != None and len(graph.sinks[node]) > 0 and isinstance(node, TileNode):
                c += node.input_port_latencies[parent.port]

            cycles.add(c)

        if None in cycles:
            cycles.remove(None)

        if len(cycles) > 1:
            print(
                f"\tWarning: Incorrect delay within kernel: {node.kernel} {node} {cycles}"
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
            print(
                f"\tWarning: Incorrect delay at output of kernel: {kernel} {kernel_output_delays}"
            )

    kernel_latencies = {}
    for kernel in node_cycles:
        kernel_cycles = [cyc for cyc in node_cycles[kernel].values() if cyc != None]
        kernel_cycles.append(0)
        kernel_latencies[kernel] = max(kernel_cycles)

    return kernel_latencies


def flush_cycles(graph, harden_flush, pipeline_config_interval):
    if harden_flush:
        flush_cycles = {}
        for mem in graph.get_mems() + graph.get_ponds():
            if mem.y == 0:
                flush_cycles[mem] = 0
            else:
                flush_cycles[mem] = (mem.y - 1) // pipeline_config_interval
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


    max_flush_cycle = max(flush_cycles.values())
    for mem,flush_c in flush_cycles.items():
        flush_cycles[mem] = -(flush_c - max_flush_cycle)

    return flush_cycles


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
):
    kernel_latencies = branch_delay_match_within_kernels(
        graph, id_to_name, placement, routing
    )

    kernel_graph = construct_kernel_graph(graph, kernel_latencies)

    flush_latencies = flush_cycles(graph, harden_flush, pipeline_config_interval)

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
        pond_latencies[id_to_name[pond_node.tile_id]] = pond_node.input_port_latencies[
            "data_in_pond"
        ]

    fout = open(kernel_latencies_file, "w")
    fout.write(json.dumps(matched_kernel_latencies, indent=4))

    fout = open(flush_latencies_file, "w")
    fout.write(json.dumps(matched_flush_latencies, indent=4))

    fout = open(pond_latencies_file, "w")
    fout.write(json.dumps(pond_latencies, indent=4))


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
):
    if load_only:
        id_to_name_filename = os.path.join(app_dir, f"design.id_to_name")
        if os.path.isfile(id_to_name_filename):
            id_to_name = load_id_to_name(id_to_name_filename)
        return placement, routing, id_to_name

    if "PIPELINED" in os.environ and os.environ["PIPELINED"] == "1":
        pe_cycles = 1
    else:
        pe_cycles = 0

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
    )

    curr_freq, crit_path, crit_nets = sta(graph)

    update_kernel_latencies(
        app_dir,
        graph,
        id_to_name,
        placement,
        routing,
        harden_flush,
        pipeline_config_interval,
    )

    freq_file = os.path.join(app_dir, "design.freq")
    fout = open(freq_file, "w")
    fout.write(f"{curr_freq}\n")

    dump_id_to_name(app_dir, id_to_name)

    return placement, routing, id_to_name
