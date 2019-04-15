import os
import subprocess
import pycyclone


def route(packed_filename: str, placement_filename,
          graph_path: str, route_result: str):
    tokens = graph_path.split()
    assert len(tokens) == 2
    # input checking
    for idx in range(0, len(tokens), 2):
        assert tokens[idx].isdigit()
        graph_path = tokens[idx + 1]
        assert os.path.isfile(graph_path)

    path = os.path.abspath(os.path.dirname(pycyclone.__file__))
    router_binary = os.path.join(path, "router")
    assert os.path.isfile(router_binary), router_binary + "not found"

    subprocess.check_call([router_binary, packed_filename, placement_filename,
                           graph_path, route_result])
