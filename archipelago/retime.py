import pprint
from typing import Dict, List


def get_new_reg(id_to_name):
    reg_id = len(id_to_name) + 1
    blk_id = "r" + str(reg_id)
    blk_name = "REG_{0}".format(reg_id)
    id_to_name[blk_id] = blk_name
    return blk_id


def get_new_net_id(netlist):
    net_id = len(netlist) + 1
    return "e" + str(net_id)


def get_register_inputs(netlist, id_to_name, bus_width):
    new_nets = {}
    for net_id, net in netlist.items():
        src_pin = net[0]
        width = 0
        if src_pin[0][0] == "i":
            width = 1
        elif src_pin[0][0] == "I":
            width = 16
        if width > 0:
            new_net = net[1:]
            new_reg = get_new_reg(id_to_name)
            new_pin = (new_reg, "reg")
            new_net = [new_pin] + new_net
            net.clear()
            net.append(src_pin)
            net.append(new_pin)
            net_id = get_new_net_id(netlist)
            new_nets[net_id] = new_net
            bus_width[net_id] = width
    netlist.update(new_nets)


class Node:
    def __init__(self, blk_id: str):
        self.blk_id = blk_id
        # next and prev are pin: net_id. net_id are always up-to-date
        self.next = {}
        self.prev = {}

        self.next_nodes = []

    def __hash__(self):
        return hash(self.blk_id)

    def __eq__(self, other):
        if not isinstance(other, Node): return False
        return self.blk_id == other.blk_id

    def __repr__(self):
        return self.blk_id


class Graph:
    def __init__(self, netlist, id_to_name, bus_width):
        self.nodes: Dict[str, Node] = {}
        self.netlist = netlist
        self.id_to_name = id_to_name
        self.bus_width = bus_width

        for net_id, net in netlist.items():
            src_pin = net[0]
            src_blk_id, src_port = src_pin
            src_node = self.__get_node(src_pin)
            src_node.next[src_port] = net_id
            for sink in net[1:]:
                sink_blk, sink_port = sink
                sink_node = self.__get_node(sink)
                sink_node.prev[sink_port] = net_id
                src_node.next_nodes.append(sink_node)
            if src_port in {"io2f_1", "alu_res_p", "valid"}:
                bus_width[net_id] = 1
            else:
                bus_width[net_id] = 16

    def __get_node(self, pin) -> Node:
        node, port = pin
        if node not in self.nodes:
            n = Node(node)
            self.nodes[node] = n
        return self.nodes[node]

    def retime(self):
        nodes = self.__sort()
        wave_info = self.__initialize_wave_info()
        for node in nodes:
            # wave matching first
            blk_type = node.blk_id[0]
            if blk_type in {"r", "c", "C", "I", "i"}:
                continue
            if blk_type == "m":
                # memory only needs wave matching
                self.__wave_matching(node, wave_info)
            else:
                assert blk_type == "p"
                # wave matching first
                self.__wave_matching(node, wave_info)
                # then input a register for the output
                sink_ports = list(node.next.keys())
                sink_ports.sort()
                for sink_port in sink_ports:
                    sink_net_id = node.next[sink_port]
                    sink_pin = (node.blk_id, sink_port)
                    new_net_id = self.__insert_pipeline_reg(sink_pin, sink_net_id, wave_info)
                    node.next[sink_port] = new_net_id

        pprint.pprint(nodes)

    @staticmethod
    def __sort_helper(node: Node, visited, stack):
        visited.add(node)

        for n in node.next_nodes:
            if n not in visited:
                Graph.__sort_helper(n, visited, stack)
        stack.append(node)

    def __sort(self):
        visited = set()
        stack = []
        for n in self.nodes.values():
            if n not in visited:
                Graph.__sort_helper(n, visited, stack)
        return stack[::-1]

    def __initialize_wave_info(self):
        # we calculate the src wave information
        result = {}
        for blk_id, node in self.nodes.items():
            for src_port in node.prev.keys():
                result[(blk_id, src_port)] = 0
        return result

    def __wave_matching(self, node, wave_info):
        max_wave = 0
        # first pass to determine the max wave
        for src_port in node.prev.keys():
            pin = (node.blk_id, src_port)
            wave = wave_info[pin]
            if wave > max_wave:
                max_wave = wave

        ports = list(node.prev.keys())
        ports.sort()
        for src_port in ports:
            pin = (node.blk_id, src_port)
            wave = wave_info[pin]
            if wave < max_wave:
                # insert registers
                for i in range(max_wave - wave):
                    net_id = node.prev[src_port]
                    self.__insert_pipeline_reg(pin, net_id, wave_info)

        # sanity check
        for src_port in node.prev.keys():
            pin = (node.blk_id, src_port)
            wave = wave_info[pin]
            assert wave == max_wave

        # update sink pin info
        for sink_port in node.next.keys():
            pin = (node.blk_id, sink_port)
            wave_info[pin] = max_wave

    def __insert_pipeline_reg(self, pin, net_id, wave_info) -> str:
        net: List = self.netlist[net_id]
        if net[0] == pin:
            # it's a source
            src_pin = pin
            new_net_id = get_new_net_id(self.netlist)
            new_reg_id = get_new_reg(self.id_to_name)
            new_reg = (new_reg_id, "reg")
            net[0] = new_reg
            new_net = [src_pin, new_reg]
            self.netlist[new_net_id] = new_net
            self.bus_width[new_net_id] = self.bus_width[net_id]
            # every source pin in the net has to increase its wave number
            for sink_pin in net[1:]:
                wave_info[sink_pin] = wave_info[pin] + 1

            node = self.__get_node(pin)
            port = pin[-1]
            assert port in node.next
            node.next[pin[-1]] = new_net_id
        else:
            # it's a sink
            # if it's a constant, we do nothing
            if net[0][0][0] in {"c", "C"}:
                # unchanged
                new_net_id = net_id
            else:
                idx = net.index(pin)
                new_reg_id = get_new_reg(self.id_to_name)
                new_reg = (new_reg_id, "reg")
                net[idx] = new_reg
                new_net_id = get_new_net_id(self.netlist)
                new_net = [new_reg, pin]
                self.netlist[new_net_id] = new_net
                self.bus_width[new_net_id] = self.bus_width[net_id]
                node = self.__get_node(pin)
                port = pin[-1]
                assert port in node.prev
                node.prev[pin[-1]] = new_net_id
            # only increase the pin wave number
            wave_info[pin] += 1

        return new_net_id


def retime_netlist(netlist, id_to_name):
    # register input first
    bus_width = {}
    get_register_inputs(netlist, id_to_name, bus_width)
    # construct the graph
    g = Graph(netlist, id_to_name, bus_width)

    g.retime()
    return g.bus_width


def netlist_to_dot(netlist, filename):
    with open(filename, "w+") as f:
        f.write("digraph netlist {\n  node [shape=box];\n")
        for net in netlist.values():
            src_id = net[0][0]
            for sink_id, port in net[1:]:
                f.write("  \"{0}\" -> \"{1}\"\n".format(src_id, sink_id))
        f.write("}\n")


def main():
    netlist = {
        "e0": [("I0", "io2f_16"), ("p0", "data0"), ("m0", "data_in_0")],
        "e1": [("p0", "alu_res"), ("p1", "data0")],
        "e2": [("m0", "data_out_0"), ("p2", "data1")],
        "e3": [("c0", "out"), ("p1", "data1")],
        "e4": [("p1", "alu_res"), ("p3", "data0")],
        "e5": [("p2", "alu_res"), ("p3", "data1"), ("p4", "data1")],
        "e6": [("p3", "alu_res"), ("p4", "data0")],
        "e7": [("p4", "alu_res"), ("I1", "f2io_16")]
    }
    netlist_to_dot(netlist, "before.dot")
    id_to_name = {"I0": "I0", "m0": "m0", "p0": "p0", "p1": "p1", "p2": "p2", "c0": "c0", "I1": "I1"}
    bus_width = retime_netlist(netlist, id_to_name)

    pprint.pprint(bus_width)
    netlist_to_dot(netlist, "after.dot")


if __name__ == "__main__":
    main()
