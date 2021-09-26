import os
import subprocess
import pycyclone


def route(packed_filename: str, placement_filename,
          graph_paths: str, route_result: str,
          max_frequency, layout=None, wave_info=None,
          shift_registers=False):
    # check input
    tokens = graph_paths.split()
    assert len(tokens) % 2 == 0
    # input checking
    graph_flags = []
    for idx in range(0, len(tokens), 2):
        assert tokens[idx].isdigit()
        graph_path = tokens[idx + 1]
        assert os.path.isfile(graph_path)
        graph_flags += ["-g", graph_path]

    path = os.path.abspath(os.path.dirname(pycyclone.__file__))
    router_binary = os.path.join(path, "router")
    assert os.path.isfile(router_binary), router_binary + "not found"
    args = [router_binary, "-p", packed_filename, "-P", placement_filename] + graph_flags + \
           ["-o", route_result]
    # timing
    if max_frequency is not None:
        assert os.path.exists(layout)
        # need to load layout as well
        args += ["-f", str(max_frequency), "-t", "default", "-l", layout]
        if wave_info is not None:
            args += ["-w", wave_info]
    elif shift_registers:
        assert os.path.exists(layout)
        args += ["-t", "register", "-l", layout]
    subprocess.check_call(args)
