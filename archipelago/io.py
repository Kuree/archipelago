def dump_packing_result(netlist, filename):
    def tuple_to_str(t_val):
        return "(" + ", ".join([str(val) for val in t_val]) + ")"
    # netlists
    with open(filename, "w+") as f:
        f.write("Netlists:\n")
        net_ids = list(netlist.keys())
        net_ids.sort(key=lambda x: int(x[1:]))
        for net_id in net_ids:
            f.write("{}: ".format(net_id))
            f.write("\t".join([tuple_to_str(entry)
                               for entry in netlist[net_id]]))

    f.write("ID to Names:\n")
    ids = set()
    for _, net in netlist.items():
        for blk_id in net:
            ids.add(blk_id)
    ids = list(ids)
    ids.sort(key=lambda x: int(x[1:]))
    for blk_id in ids:
        f.write(str(blk_id) + ": " + str(blk_id) + "\n")


def load_routing_result(filename):
    # copied from pnr python implementation
    with open(filename) as f:
        lines = f.readlines()

    routes = {}
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        line_index += 1
        if line[:3] == "Net":
            tokens = line.split(" ")
            net_id = tokens[2]
            routes[net_id] = []
            num_seg = int(tokens[-1])
            for seg_index in range(num_seg):
                segment = []
                line = lines[line_index].strip()
                line_index += 1
                assert line[:len("Segment")] == "Segment"
                tokens = line.split()
                seg_size = int(tokens[-1])
                for i in range(seg_size):
                    line = lines[line_index].strip()
                    line_index += 1
                    line = "".join([x for x in line if x not in ",()"])
                    tokens = line.split()
                    tokens = [int(x) if x.isdigit() else x for x in tokens]
                    segment.append(tokens)
                routes[net_id].append(segment)
    return routes