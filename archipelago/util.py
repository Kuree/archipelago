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
