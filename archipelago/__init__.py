import tempfile
import os
import shutil
from .io import dump_packing_result, load_routing_result, dump_placement_result
from .util import parse_routing_result
from .place import place
from .route import route
import pycyclone


def pnr(arch, input_netlist=None, packed_file="", cwd="", app_name=""):
    if input_netlist is None and len(packed_file):
        raise ValueError("both input")

    use_temp = False
    app_name = "design" if len(app_name) == 0 else app_name

    if len(cwd) == 0:
        # get a temp cwd
        use_temp = True
        cwd = tempfile.TemporaryDirectory()

    if not isinstance(arch, str):
        # attempt to treat it as an interconnect object
        if hasattr(arch, "dump_pnr"):
            # use dump pnr instead
            arch.dump_pnr(cwd, "design")
            arch_file = os.path.join(cwd, "design.info")
        else:
            raise Exception("arch has to be either string or interconnect")
    else:
        arch_file = arch

    # prepare for the netlist
    if len(packed_file) == 0:
        packed_file = dump_packed_result(app_name, cwd, input_netlist)

    # get the layout and routing file
    with open(arch_file) as f:
        layout_line = f.readline()
        layout_filename = layout_line.split("=")[-1].strip()
        assert os.path.isfile(layout_filename)
        graph_path_line = f.readline()
        graph_path = graph_path_line.split("=")[-1].strip()

    # do the place and route
    placement_filename = os.path.join(cwd, app_name + ".place")
    place(packed_file, layout_filename, placement_filename)
    route_filename = os.path.join(cwd, app_name + ".route")
    route(packed_file, placement_filename, graph_path, route_filename)

    # need to load it back up
    placement_result = pycyclone.io.load_placement(placement_filename)
    routing_result = load_routing_result(route_filename)

    # tear down
    if use_temp:
        if os.path.isdir(cwd):
            shutil.rmtree(cwd)

    if hasattr(arch, "dump_pnr"):
        routing_result = parse_routing_result(routing_result, arch)

    return placement_result, routing_result


def dump_packed_result(app_name, cwd, inputs):
    assert inputs is not None
    input_netlist, input_bus = inputs
    assert isinstance(input_netlist, dict)
    netlist = {}
    for net_id, net in input_netlist.items():
        assert isinstance(net, list)
        for entry in net:
            assert len(entry) == 2, "entry in the net has to be " \
                                    "(blk_id, port)"
        netlist[net_id] = net
    # dump the packed file
    packed_file = os.path.join(cwd, app_name + ".packed")
    dump_packing_result(netlist, input_bus, packed_file)
    return packed_file
