import os
import sys
import pycyclone
import glob
import csv
import argparse

from .io import load_routing_result


def load_graph(graph_files):
    graph_result = {}
    for graph_file in graph_files:
        bit_width = os.path.splitext(graph_file)[0]
        bit_width = int(os.path.basename(bit_width))
        graph = pycyclone.io.load_routing_graph(graph_file)
        graph_result[bit_width] = graph
    return graph_result


def get_all_sb_in_graph(graphs):
    # indexed by coord, then width
    result = {}
    for width, graph in graphs.items():
        for x, y in graph:
            if (x, y) not in result:
                result[x, y] = {}
            tile = graph[x, y]
            switchbox = tile.switchbox
            sides = switchbox.SIDES
            tile_sbs = []
            for i in range(sides):
                side = pycyclone.SwitchBoxSide(i)
                sbs = switchbox.get_sbs_by_side(side)
                tile_sbs += sbs
            result[x, y][width] = tile_sbs

    return result


def get_used_sbs(routing_result):
    # indexed by coord
    result = {}
    for netlist in routing_result.values():
        for net in netlist:
            for node in net:
                if node[0] == "SB":
                    # need to parse the SB logic
                    track, x, y, side, io_, bit_width = node[1:]
                    entry = (track, side, io_, bit_width)
                    if (x, y) not in result:
                        result[x, y] = set()
                    result[x, y].add(entry)
    return result


def compute_usage(graph_sbs, routing_sbs):
    # result is indexed by (x, y), then bit_width, and then in/out
    total = {}
    result = {}
    # go through the graph_abs to count the stuff
    for (x, y), width_sbs in graph_sbs.items():
        total[(x, y)] = {}
        result[(x, y)] = {}
        for sbs in width_sbs.values():
            for sb in sbs:
                bit_width = sb.width
                if bit_width not in total[(x, y)]:
                    total[(x, y)][bit_width] = {}
                    result[(x, y)][bit_width] = {}
                io = sb.io
                if io not in total[(x, y)][bit_width]:
                    total[(x, y)][bit_width][io] = 0
                    result[(x, y)][bit_width][io] = 0
                total[(x, y)][bit_width][io] += 1

    # now deduct the actual usage
    for (x, y), sb_data in routing_sbs.items():
        entry = result[(x, y)]
        for track, side, io_, bit_width in sb_data:
            entry[bit_width][pycyclone.SwitchBoxIO(io_)] += 1

    return result, total


def produce_stats(used_sbs, total_sbs, filename):
    header = ["X", "Y", "TRACK_WIDTH", "IO", "USED", "TOTAL"]
    rows = []
    for (x, y), entry in used_sbs.items():
        for bit_width, io_entry in entry.items():
            for io_, value in io_entry.items():
                used_value = value
                total_value = total_sbs[(x, y)][bit_width][io_]
                rows.append([x, y, bit_width, io_.name, used_value, total_value])
    result = [header] + rows
    with open(filename, "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(result)


def main():
    parser = argparse.ArgumentParser("PnR Routing Analysis Tool")
    parser.add_argument("-i,-d", action="store", type=str, required=True, dest="input_dir",
                        help="PnR collateral directory")
    parser.add_argument("-o,-f", action="store", type=str, required=True, dest="csv_filename",
                        help="Output CSV filename")
    args = parser.parse_args()
    data_dir = args.input_dir
    assert os.path.isdir(data_dir), data_dir + " is not a directory"
    files = glob.glob(os.path.join(data_dir, "*"))
    assert len(files) > 0, data_dir + " is empty"
    # find out useful files
    graph_files = [x for x in files if os.path.splitext(x)[-1] == ".graph"]
    assert len(graph_files) > 0, "Unable to find any routing graph from " + data_dir
    route_file = [x for x in files if os.path.splitext(x)[-1] == ".route"]
    assert len(route_file) == 1, "Only one PnR route result can be in the data folder"
    route_file = route_file[0]

    # load the graph
    graphs = load_graph(graph_files)

    # load raw routing result
    raw_routing_result = load_routing_result(route_file)

    # get all sbs
    available_sbs = get_all_sb_in_graph(graphs)

    used_sbs = get_used_sbs(raw_routing_result)

    result_sbs, total_sbs = compute_usage(available_sbs, used_sbs)

    produce_stats(result_sbs, total_sbs,  args.csv_filename)


if __name__ == "__main__":
    main()
