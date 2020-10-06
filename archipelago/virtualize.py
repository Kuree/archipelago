import subprocess
import tempfile
from .pnr_ import pnr
from .io import dump_packed_result, load_packing_result
import os
import pythunder
from .util import get_total_clb_per_group, get_group_size, get_max_num_col


def partition(inputs, id_to_name, max_partition_size, cwd):
    # notice that this will set IO locations as well
    # dump the netlist somewhere so that it can read by the partition tool
    packed_file = dump_packed_result("design", cwd, inputs, id_to_name)
    # call the partition binary
    args = ["partition", packed_file, cwd, "-m", str(max_partition_size)]
    print(args)
    subprocess.check_call(args)

    # need to load up every partitioned design packed and design place
    result = {}
    partition_id = 0
    while True:
        packed_file = os.path.join(cwd, str(partition_id) + ".packed")
        if not os.path.isfile(packed_file):
            break
        placement_file = os.path.join(cwd, str(partition_id) + ".place")
        assert os.path.isfile(placement_file), "Unable to find " + placement_file

        io_placement = pythunder.io.load_placement(placement_file)

        result[partition_id] = (packed_file, io_placement)

        partition_id += 1

    assert len(result) > 0, "Unable to partition the netlist"

    return result


def pnr_virtualize(interconnect, inputs, cwd, id_to_name, max_group):
    # need to compute the max_size based on max_column
    # use PE as based count
    assert hasattr(interconnect, "dump_pnr"), "Only canal arch currently supported"
    group_size = get_group_size(interconnect)
    total_clb = get_total_clb_per_group(interconnect, group_size)
    partition_result = partition(inputs, id_to_name, total_clb * max_group, cwd)
    num_col = max_group * group_size

    # for each partition, we need to call pnr function on each partition
    result = {}
    for cluster_id, (packed_file, io_placement) in partition_result.items():
        # need to sanity check
        cluster_netlist = load_packing_result(packed_file)[0][0]
        max_cluster_column = get_max_num_col(cluster_netlist, interconnect)
        assert max_cluster_column <= num_col,\
            "Cannot fit netlist inside " + str(max_group) + " groups"
        partition_id_to_name = pythunder.io.load_id_to_name(packed_file)
        cluster_result = pnr(interconnect, inputs, packed_file=packed_file,
                             cwd=cwd, app_name=str(cluster_id),
                             id_to_name=partition_id_to_name,
                             fixed_pos=io_placement, max_num_col=num_col)
        result[cluster_id] = cluster_result, partition_id_to_name
    return result


