import tempfile
import os
from .io import dump_packed_result
from .place import place
from .route import route
from .io import dump_packing_result, load_routing_result, dump_placement_result
from .util import parse_routing_result

import pycyclone


def pnr(arch, input_netlist=None, packed_file="", cwd="", app_name="",
        id_to_name=None, fixed_pos=None, max_num_col=None):
    if input_netlist is None and len(packed_file):
        raise ValueError("Invalid input")

    use_temp = False
    app_name = "design" if len(app_name) == 0 else app_name

    if len(cwd) == 0:
        # get a temp cwd
        use_temp = True
        cwd_dir = tempfile.TemporaryDirectory()
        cwd = cwd_dir.name
    else:
        cwd_dir = None

    if not isinstance(arch, str):
        # attempt to treat it as an interconnect object
        if hasattr(arch, "dump_pnr"):
            # if virtualization is turned on with canal, we can dynamically
            # dump the adjusted size and partition
            # we assume the netlist is already partitioned
            arch.dump_pnr(cwd, "design", max_num_col=max_num_col)
            arch_file = os.path.join(cwd, "design.info")
        else:
            raise Exception("arch has to be either string or interconnect")
    else:
        arch_file = arch

    # prepare for the netlist
    if len(packed_file) == 0:
        packed_file = dump_packed_result(app_name, cwd, input_netlist,
                                         id_to_name)

    # get the layout and routing file
    with open(arch_file) as f:
        layout_line = f.readline()
        layout_filename = layout_line.split("=")[-1].strip()
        assert os.path.isfile(layout_filename)
        graph_path_line = f.readline()
        graph_path = graph_path_line.split("=")[-1].strip()

    # get placement name
    placement_filename = os.path.join(cwd, app_name + ".place")

    # if we have fixed
    if fixed_pos is not None:
        assert isinstance(fixed_pos, dict)
        dump_placement_result(fixed_pos, placement_filename, id_to_name)
        has_fixed = True
    else:
        has_fixed = False

    # do the place and route
    place(packed_file, layout_filename, placement_filename, has_fixed)
    route_filename = os.path.join(cwd, app_name + ".route")
    route(packed_file, placement_filename, graph_path, route_filename)

    # need to load it back up
    placement_result = pycyclone.io.load_placement(placement_filename)
    routing_result = load_routing_result(route_filename)

    # tear down
    if use_temp:
        if os.path.isdir(cwd):
            assert cwd_dir is not None
            cwd_dir.__exit__(None, None, None)

    if hasattr(arch, "dump_pnr"):
        routing_result = parse_routing_result(routing_result, arch)

    return placement_result, routing_result
