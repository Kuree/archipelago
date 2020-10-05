import pythunder


def get_layout(board_layout, pe_tag="p", memory_tag="m"):
    # copied from cgra_pnr
    new_layout = []
    for y in range(len(board_layout)):
        row = []
        for x in range(len(board_layout[y])):
            if board_layout[y][x] is None:
                row.append(' ')
            else:
                row.append(board_layout[y][x])
        new_layout.append(row)

    default_priority = pythunder.Layout.DEFAULT_PRIORITY
    # not the best practice to use the layers here
    # but this requires minimum amount of work to convert the old
    # code to the new codebase
    # FIXME: change the CGRA_INFO parser to remove the changes
    layout = pythunder.Layout(new_layout)
    # add a reg layer to the layout, the same as PE
    clb_layer = layout.get_layer(pe_tag)
    reg_layer = pythunder.Layer(clb_layer)
    reg_layer.blk_type = 'r'
    layout.add_layer(reg_layer, default_priority, 0)
    layout.set_priority_major(' ', 0)
    layout.set_priority_major('i', 1)
    layout.set_priority_major("I", 2)
    # memory is a DSP-type, so lower priority
    layout.set_priority_major(memory_tag, default_priority - 1)
    return layout


def parse_routing_result(raw_routing_result, interconnect):
    # in the original cyclone implementation we don't need this
    # since it just translate this IR into bsb format without verifying the
    # connectivity. here, however, we need to since we're producing bitstream
    result = {}
    for net_id, raw_routes in raw_routing_result.items():
        result[net_id] = []
        for raw_segment in raw_routes:
            segment = []
            for node_str in raw_segment:
                node = interconnect.parse_node(node_str)
                segment.append(node)
            result[net_id].append(segment)
    return result


def get_max_num_col(netlist, interconnect):
    # first we need to build a map of each resources
    # we assume registers are abundant
    resources = {}
    for x in range(interconnect.x_max + 1):
        resources[x] = {}
        for y in range(interconnect.y_max + 1):
            tile_circuit = interconnect.tile_circuits[(x, y)]
            pnr_tag = tile_circuit.core.pnr_info()
            tag = pnr_tag.tag_name
            if tag not in resources[x]:
                resources[x][tag] = 0
            resources[x][tag] += 1
    # compute the block resource
    required_blks = {}
    for net in netlist.values():
        for blk, _ in net:
            blk_type = blk[0]
            if blk_type == 'r':
                # we always have a lot of registers on the fabric
                continue
            if blk_type not in required_blks:
                required_blks[blk_type] = 0
            required_blks[blk_type] += 1

    # figure out how many columns required
    # notice that we do that with groups of special tiles,
    # most cases, mem tile. we need to figure out the groups
    # automatically as well
    group_size = 0
    # assume it starts from 0
    tags = []
    group_size = 0
    for x in range(interconnect.x_max + 1):
        tile_circuit = interconnect.tile_circuits[(x, interconnect.y_max // 2)]
        tag = tile_circuit.core.pnr_info().tag_name
        if not tags:
            tags.append(tag)
        else:
            if tag in tags:
                tags.append(tag)
            else:
                group_size = x + 1
    assert group_size != 0, "Unable to find group size"

    current_col = -1
    for x in range(interconnect.x_max + 1):
        should_continue = False
        for num in required_blks.values():
            if num > 0:
                should_continue = True
                break
        if not should_continue:
            break
        current_col += 1

        blks = resources[current_col]
        for blk_type, value in blks.items():
            if blk_type in required_blks:
                required_blks[blk_type] -= value

    # compute the group number required
    import math
    max_num_col = int(math.ceil(current_col / group_size)) * group_size
    return max_num_col
