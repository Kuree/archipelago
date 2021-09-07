import pprint
from typing import Dict, List


def get_new_reg(id_to_name):
    start = len(id_to_name)
    while True:
        reg_id = start
        blk_id = "r" + str(reg_id)
        if blk_id not in id_to_name:
            break
        else:
            start += 1
    blk_name = "REG_{0}".format(reg_id)
    id_to_name[blk_id] = blk_name
    return blk_id


def get_new_net_id(netlist):
    start = len(netlist)
    while True:
        net_id = "e" + str(start)
        if net_id not in netlist:
            return net_id
        else:
            start += 1


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

        self.const_nodes = self.__get_const_nodes()
        self.constant_pins = self.__get_const_connected_pins()

        # some pins will be ignored
        self.__ignore_pins = {"flush"}

    def __get_node(self, pin) -> Node:
        node, port = pin
        if node not in self.nodes:
            n = Node(node)
            self.nodes[node] = n
        return self.nodes[node]

    def retime(self, save_dot=False):
        if save_dot:
            netlist_to_dot(self.netlist, "before.dot")
        nodes = self.__sort()
        wave_info = self.__initialize_wave_info()
        for node in nodes:
            # skip constant nodes
            if node in self.const_nodes:
                continue
            # wave matching first
            blk_type = node.blk_id[0]
            if blk_type in {"r", "c", "C", "I", "i"}:
                # insert a reg if not done so
                if blk_type in {"i", "I"} and len(node.next) == 0:
                    assert len(node.prev) == 1
                    for src_port, net_id in node.prev.items():
                        src_pin = (node.blk_id, src_port)
                        self.__insert_pipeline_reg(src_pin, net_id, wave_info)
                        break
                elif blk_type in {"i", "I"} and len(node.prev) == 0:
                    assert len(node.next) == 1
                    for sink_port, net_id in node.next.items():
                        sink_pin = (node.blk_id, sink_port)
                        self.__insert_pipeline_reg(sink_pin, net_id, wave_info)
                        break
                elif blk_type == "r":
                    # we need to update the dst wave number, which are the same as the source
                    self.__pass_wave_number(node, wave_info)
                continue
            if blk_type == "m":
                # memory only needs wave matching
                self.__wave_matching(node, wave_info)
                self.__pass_wave_number(node, wave_info)
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
                    self.__insert_pipeline_reg(sink_pin, sink_net_id, wave_info)

        self.__optimize_for_packing()

        if save_dot:
            netlist_to_dot(self.netlist, "after.dot")

        return Graph.__compute_final_wave_number(wave_info)

    @staticmethod
    def __compute_final_wave_number(wave_info):
        result = {}
        for pin, wave in wave_info.items():
            blk_id = pin[0]
            if blk_id not in result:
                result[blk_id] = wave
            elif result[blk_id] < wave:
                result[blk_id] = wave
        return result

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
            if blk_id[0] in {"i", "I"} and len(node.prev) == 0:
                for sink_port in node.next.keys():
                    result[(blk_id, sink_port)] = 0
        return result

    def __wave_matching(self, node, wave_info):
        max_wave = 0
        # first pass to determine the max wave
        for src_port in node.prev.keys():
            pin = (node.blk_id, src_port)
            if pin in self.constant_pins or src_port in self.__ignore_pins:
                continue
            wave = wave_info[pin]
            if wave > max_wave:
                max_wave = wave

        ports = list(node.prev.keys())
        ports.sort()
        for src_port in ports:
            pin = (node.blk_id, src_port)
            if pin in self.constant_pins or src_port in self.__ignore_pins:
                continue
            wave = wave_info[pin]
            if wave < max_wave:
                # insert registers
                for i in range(max_wave - wave):
                    net_id = node.prev[src_port]
                    self.__insert_pipeline_reg(pin, net_id, wave_info)

        # sanity check
        for src_port in node.prev.keys():
            pin = (node.blk_id, src_port)
            if pin in self.constant_pins or src_port in self.__ignore_pins:
                continue
            wave = wave_info[pin]
            assert wave == max_wave

        # update sink pin info
        for sink_port in node.next.keys():
            pin = (node.blk_id, sink_port)
            wave_info[pin] = max_wave

    def __pass_wave_number(self, node, wave_info):
        # we assume the wave number already match
        max_wave_num = 0
        for src_port in node.prev.keys():
            pin = (node.blk_id, src_port)
            wave_num = wave_info[pin]
            if max_wave_num < wave_num:
                max_wave_num = wave_num

        for net_id in node.next.values():
            net = self.netlist[net_id]
            for pin in net[1:]:
                wave_info[pin] = max_wave_num

    def __insert_pipeline_reg(self, pin, net_id, wave_info) -> str:
        net: List = self.netlist[net_id]
        if net[0] == pin:
            # it's a source
            src_pin = pin
            new_net_id = get_new_net_id(self.netlist)
            new_reg_id = get_new_reg(self.id_to_name)
            new_reg_out = (new_reg_id, "out")
            new_reg_in = (new_reg_id, "in")
            net[0] = new_reg_out
            new_net = [src_pin, new_reg_in]
            self.netlist[new_net_id] = new_net
            self.bus_width[new_net_id] = self.bus_width[net_id]
            # every source pin in the net has to increase its wave number
            for sink_pin in net[1:]:
                wave_info[sink_pin] = wave_info[pin] + 1

            node = self.__get_node(pin)
            port = pin[-1]
            assert port in node.next
            node.next[pin[-1]] = new_net_id

            # handle new node info
            new_reg_node = self.__get_node(new_reg_in)
            new_reg_node.prev["in"] = new_net_id
            new_reg_node.next["out"] = net_id
        else:
            # it's a sink
            # if it's a constant, we do nothing
            if net[0][0][0] in {"c", "C"}:
                # unchanged
                new_net_id = net_id
            else:
                idx = net.index(pin)
                new_reg_id = get_new_reg(self.id_to_name)
                new_reg_in = (new_reg_id, "in")
                new_reg_out = (new_reg_id, "out")
                net[idx] = new_reg_in
                new_net_id = get_new_net_id(self.netlist)
                new_net = [new_reg_out, pin]
                self.netlist[new_net_id] = new_net
                self.bus_width[new_net_id] = self.bus_width[net_id]
                node = self.__get_node(pin)
                port = pin[-1]
                assert port in node.prev
                node.prev[pin[-1]] = new_net_id

                # handle new node info
                new_reg_node = self.__get_node(new_reg_in)
                new_reg_node.prev["in"] = net_id
                new_reg_node.next["out"] = new_net_id
            # only increase the pin wave number
            wave_info[pin] += 1

        return new_net_id

    def __get_const_nodes(self):
        # constant nodes are nodes that's not IO but doesn't have source
        result = set()
        for node in self.nodes.values():
            if node.blk_id[0] not in {"i", "I"} and len(node.prev) == 0:
                result.add(node)
        return result

    def __get_const_connected_pins(self):
        result = set()
        for node in self.const_nodes:
            for net_id in node.next.values():
                net = self.netlist[net_id]
                for pin in net[1:]:
                    result.add(pin)
        return result

    def __optimize_for_packing(self):
        # optimize the netlist for packing
        # we split registers to each PE operands
        net_ids = list(self.netlist.keys())
        net_ids.sort()
        nets_to_remove = set()
        for net_id in net_ids:
            net = self.netlist[net_id]
            if len(net) <= 2:
                continue
            if net[0][0][0] != 'r':
                continue
            match = True
            for pin in net[1:]:
                if pin[0][0][0] != 'p':
                    match = False
                    break
            if not match:
                continue
            # now insert another pipeline registers
            src_pin = net[0]
            renamed_src_pin = (src_pin[0], "in")
            src_node = self.__get_node(src_pin)
            src_net_id = src_node.prev["in"]
            src_net = self.netlist[src_net_id]
            # remove the source pin
            src_net.remove(renamed_src_pin)
            for sink_pin in net[1:]:
                sink_node = self.__get_node(sink_pin)
                new_reg_id = get_new_reg(self.id_to_name)
                new_reg_pin_in = (new_reg_id, "in")
                new_reg_pin_out = (new_reg_id, "out")
                new_net = [new_reg_pin_out, sink_pin]
                new_net_id = get_new_net_id(self.netlist)
                self.netlist[new_net_id] = new_net
                self.bus_width[new_net_id] = self.bus_width[net_id]
                sink_node.prev[sink_pin[-1]] = new_net_id
                # add the new reg pin to the original net
                src_net.append(new_reg_pin_in)

            nets_to_remove.add(net_id)

        for net_id in nets_to_remove:
            del self.netlist[net_id]


def retime_netlist(netlist, id_to_name, bus_width, type_printout=None):
    # register input first
    # get_register_inputs(netlist, id_to_name, bus_width)
    # construct the graph
    g = Graph(netlist, id_to_name, bus_width)

    wave = g.retime()
    if type_printout is not None:
        print("Wave info:")
        for node, w in wave.items():
            blk_type = node[0]
            if blk_type in type_printout:
                print(id_to_name[node] + ":", w)
    return g.bus_width


def netlist_to_dot(netlist, filename):
    with open(filename, "w+") as f:
        f.write("digraph netlist {\n  node [shape=box];\n")
        for net in netlist.values():
            src_id = net[0][0]
            for sink_id, port in net[1:]:
                f.write("  \"{0}\" -> \"{1}\"\n".format(src_id, sink_id))
        f.write("}\n")


def load_packing_result(filename):
    import pythunder
    netlist, bus_mode = pythunder.io.load_netlist(filename)
    id_to_name = pythunder.io.load_id_to_name(filename)
    return netlist, id_to_name, bus_mode


def load_data_from_json(filename):
    import json
    with open(filename) as f:
        data = json.load(f)
    raw_netlist = data["netlist"]
    netlist = {}
    for net_id in raw_netlist:
        net = raw_netlist[net_id]
        new_net = []
        for pin in net:
            new_net.append(tuple(pin))
        netlist[net_id] = new_net
    return netlist, data["id_to_name"], data["bus_width"]


def main():
    # netlist, id_to_name, bus_width = load_packing_result("gaussian.packed")
    netlist, id_to_name, bus_width = load_data_from_json("data.json")
    netlist_to_dot(netlist, "before.dot")
    pprint.pprint(netlist)
    retime_netlist(netlist, id_to_name, bus_width=bus_width, type_printout="iIm")

    pprint.pp(netlist)
    netlist_to_dot(netlist, "after.dot")


if __name__ == "__main__":
    main()
