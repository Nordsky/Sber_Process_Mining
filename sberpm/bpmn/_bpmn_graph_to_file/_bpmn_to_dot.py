# Pydotplus Python module is used in this file.
#   Licence: MIT License
#   Link: https://github.com/carlos-jenkins/pydotplus

from ...visual._types import NodeType

import pydotplus


def bpmn_to_graph(bpmn_graph):
    """
    Transform given bpmn graph to a graph object that can be visualized.

    Parameters
    ----------
    bpmn_graph: Graph
        BPMN graph.

    Returns
    ----------
    pydot_plus_graph: pydotplus.Dot
        Representation of the graph that can be visualized.
    """
    pydot_plus_graph = pydotplus.Dot()
    pydot_plus_node_id_dict = dict()
    for node in bpmn_graph.get_nodes():
        id_with_space = add_last_space(node.id)
        if node.type == NodeType.START_EVENT:
            n = pydotplus.Node(name=id_with_space, label=add_last_space(''), shape='circle', fillcolor='green')
        elif node.type == NodeType.END_EVENT:
            n = pydotplus.Node(name=id_with_space, label=add_last_space(''), shape='circle', fillcolor='red')
        elif node.type == NodeType.TASK:
            n = pydotplus.Node(name=id_with_space, label=add_last_space(node.label), shape='box')
        elif node.type == NodeType.PARALLEL_GATEWAY:
            n = pydotplus.Node(name=id_with_space, label=add_last_space('+'), shape='diamond')
        elif node.type == NodeType.EXCLUSIVE_GATEWAY:
            n = pydotplus.Node(name=id_with_space, label=add_last_space('x'), shape='diamond')
        else:
            raise TypeError(f'Node of type "{node.type}" is not expected to be in a BPMN graph.')
        pydot_plus_node_id_dict[node.id] = n
        pydot_plus_graph.add_node(n)

    for edge in bpmn_graph.get_edges():
        e = pydotplus.Edge(src=pydot_plus_node_id_dict[edge.source_node.id],
                           dst=pydot_plus_node_id_dict[edge.target_node.id])
        pydot_plus_graph.add_edge(e)

    return pydot_plus_graph


def add_last_space(s):
    return f'{s} '
