import sys


def __get_net_id(id_num, prefix="f"):
    return "{1}{0}".format(id_num, prefix)


def reduce_switching(routing_result, interconnect, ignore_tiles=None, compact=False):
    import canal
    if ignore_tiles is None:
        ignore_tiles = {}
    result = {}
    # flat used nodes
    used_nodes = set()
    for route in routing_result.values():
        for seg in route:
            for node in seg:
                used_nodes.add(node)

    # if compact mode is on, never go beyond the max x
    if compact:
        max_x = 0
        for node in used_nodes:
            if node.x > max_x:
                max_x = node.x
    else:
        max_x = 0xFFFFFFFF

    # loop through each sb node
    for tile_circuit in interconnect.tile_circuits.values():
        for tile in tile_circuit.tiles.values():
            x, y = tile.x, tile.y
            if (x, y) in ignore_tiles:
                continue
            if x > max_x:
                continue
            switch_box = tile.switchbox
            num_tracks = switch_box.num_track
            for side in canal.interconnect.SwitchBoxSide:
                for t in range(num_tracks):
                    sb = switch_box.get_sb(side, t, canal.interconnect.SwitchBoxIO.SB_OUT)
                    if sb in used_nodes:
                        continue
                    # unused
                    # check connections
                    conn_in = sb.get_conn_in()
                    if conn_in[0] not in used_nodes:
                        # no need to worry about since it's not going to be used
                        continue
                    has_fix = False
                    for node in conn_in[1:]:
                        if node not in used_nodes:
                            # we found a fix
                            net_id = __get_net_id(len(result))
                            result[net_id] = [[node, sb]]
                            has_fix = True
                            break
                    if not has_fix:
                        print("Unable to find a switching fix for", sb, file=sys.stderr)
    print("Using extra", len(result), "connections to fix the routing")
    return result


def turn_off_tiles(routing_result, interconnect):
    # final bitstream
    bitstream = []
    # get always_on tiles
    always_on_tiles = []
    for route in routing_result.values():
        for seg in route:
            for node in seg:
                always_on_tiles.append((node.x, node.y))

    always_off_tiles = []
    for (x, y), tile_circuit in interconnect.tile_circuits.items():
        # make sure that we can turn this tile off
        features = tile_circuit.features()
        for feat_addr, feat in enumerate(features):
            if "PowerDomainConfig" in feat.name():
                if (x, y) in always_on_tiles:
                    break
                else:
                    always_off_tiles.append((x, y))
                addr, data = feat.configure(True)
                addr = interconnect.get_config_addr(addr, feat_addr, x, y)
                bitstream.append((addr, data))
                break

    return bitstream, (always_on_tiles, always_off_tiles)


def fix_x(routing_result, interconnect, on_off_tiles):
    import canal

    (always_on_tiles, always_off_tiles) = on_off_tiles
    # final result
    result = {}
    # find boundary tiles
    boundary_tiles = []
    graph = interconnect.get_graph(interconnect.get_bit_widths()[0])
    size_x, size_y = graph.get_size()
    for x, y in always_on_tiles:
        x_min = max(0, x - 1)
        x_max = min(size_x - 1, x + 1)
        y_min = max(0, y - 1)
        y_max = min(size_y - 1, y + 1)
        points = [(x_min, y), (x_max, y), (x, y_min), (x, y_max)]
        for loc in points:
            if loc in always_off_tiles and loc != (x, y):
                boundary_tiles.append(loc)

    # compute all used nodes
    used_nodes = set()
    for route in routing_result.values():
        for seg in route:
            for node in seg:
                used_nodes.add((node.x, node.y))

    # for all the boundary tiles, making sure that the cb is gated, if not used
    for loc in boundary_tiles:
        tile_circuit = interconnect.tile_circuits[loc]
        # make sure that we can turn this tile off
        # for any connection box, use the 0 value at the end of each connection
        cbs = tile_circuit.cbs
        for cb in cbs.values():
            cb_node = cb.node
            if cb_node not in used_nodes:
                # gate that
                node = cb_node.get_conn_in()[-1]
                net_id = __get_net_id(len(result), "x")
                result[net_id] = [[node, cb_node]]

        # for any switch box output, if not used, connect it to the core output
        sbs = tile_circuit.sbs
        for sb_circuit in sbs.values():
            switch_box = sb_circuit.switchbox
            num_tracks = switch_box.num_track
            for side in canal.interconnect.SwitchBoxSide:
                for t in range(num_tracks):
                    sb = switch_box.get_sb(side, t, canal.interconnect.SwitchBoxIO.SB_OUT)
                    if sb in used_nodes:
                        continue
                    nodes = sb.get_conn_in()
                    for n in nodes:
                        if isinstance(n, canal.cyclone.PortNode):
                            net_id = __get_net_id(len(result), "x")
                            result[net_id] = [[n, sb]]
                            break

    return result
