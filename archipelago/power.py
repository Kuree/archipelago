def __get_net_id(id_num):
    return "f{0}".format(id_num)


def reduce_switching(routing_result, interconnect):
    import canal
    result = {}
    # flat used nodes
    used_nodes = set()
    for route in routing_result.values():
        for seg in route:
            for node in seg:
                used_nodes.add(node)

    # loop through each sb node
    for tile_circuit in interconnect.tile_circuits.values():
        for tile in tile_circuit.tiles.values():
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
                    assert has_fix, "Unable to find a switching fix"
    print("Using extra", len(result), "connections to fix the routing")
    return result


def turn_off_tiles(routing_result, interconnect):
    # final bitstream
    bitstream = []
    # get always_on tiles
    always_on_tiles = set()
    for route in routing_result.values():
        for seg in route:
            for node in seg:
                always_on_tiles.add((node.x, node.y))

    always_off_tiles = set()
    for (x, y), tile_circuit in interconnect.tile_circuits.items():
        # make sure that we can turn this tile off
        features = tile_circuit.features()
        for feat_addr, feat in enumerate(features):
            if "PowerDomainConfig" in feat.name():
                if (x, y) in always_on_tiles:
                    break
                else:
                    always_off_tiles.add((x, y))
                addr, data = feat.configure(True)
                addr = interconnect.get_config_addr(addr, feat_addr, x, y)
                bitstream.append((addr, data))
                break

    return bitstream, (always_on_tiles, always_off_tiles)
