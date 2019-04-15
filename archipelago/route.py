import os
import subprocess
import pycyclone


def route(packed_filename: str, placement_filename,
          graph_paths: str, route_result: str):
    # check input
    tokens = graph_paths.split()
    assert len(tokens) % 2 == 0
    # input checking
    for idx in range(0, len(tokens), 2):
        assert tokens[idx].isdigit()
        graph_path = tokens[idx + 1]
        assert os.path.isfile(graph_path)

    path = os.path.abspath(os.path.dirname(pycyclone.__file__))
    router_binary = os.path.join(path, "router")
    assert os.path.isfile(router_binary), router_binary + "not found"
    args = [router_binary, packed_filename, placement_filename] + tokens + \
           [route_result]
    subprocess.check_call(args)
