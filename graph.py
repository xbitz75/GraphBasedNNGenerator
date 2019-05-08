import collections

import networkx as nx
import matplotlib.pyplot as plt


def generateGraph(type, number_of_nodes, number_of_neighbors, probability):
    if type == "WS":
        print("Generating Watts-Strogatz graph")
        return nx.generators.random_graphs.connected_watts_strogatz_graph(number_of_nodes, number_of_neighbors,
                                                                          probability)
    elif type == "BA":
        print("generating Barabási–Albert graph")
        return nx.generators.random_graphs.barabasi_albert_graph()
    elif type == "ER":
        print("generating Erdős-Rényi graph")
        return nx.generators.erdos_renyi_graph(number_of_nodes, probability)


def drawGraph(graph):
    plt.subplot(121)
    nx.draw_networkx(graph)
    #plt.savefig("graph.pdf") // saves graph as pdf file
    plt.show()


def getGraphInfo(graph):
    print(nx.info(graph))
    print("Number of selfloops: {}".format(nx.number_of_selfloops(graph)))


Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])


def _parseGraph(graph):
    _input_nodes = []
    _output_nodes = []
    _Nodes = []
    for node in range(graph.number_of_nodes()):
        _tmp = list(graph.neighbors(node))
        _tmp.sort()
        _type = "hidden_node"
        if node < _tmp[0]:
            _input_nodes.append(node)
            _type = "input_node"
        if node > _tmp[-1]:
            _output_nodes.append(node)
            _type = "output_node"
        _Nodes.append(Node(node, [n for n in _tmp if n < node], _type))
    return _Nodes, _input_nodes, _output_nodes


def getNodes(graph):
    return _parseGraph(graph)[0]


def getInputNodes(graph):
    return _parseGraph(graph)[1]


def getOutputNodes(graph):
    return _parseGraph(graph)[2]


def graphToString(graph):
    _parsed_graph = _parseGraph(graph)
    print("input nodes: {}".format(_parsed_graph[1]))
    print("output nodes: {}".format(_parsed_graph[2]))
    for node in _parsed_graph[0]:
        print(node)


# def main():
#     g = generateGraph("WS", 20, 4, 0.5)
#     getGraphInfo(g)
#     graphToString(g)
#     drawGraph(g)
#
#
# if __name__ == '__main__':
#     main()
