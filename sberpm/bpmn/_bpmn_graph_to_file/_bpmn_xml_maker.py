# Pydotplus Python module is used in this file.
#   Licence: MIT License
#   Link: https://github.com/carlos-jenkins/pydotplus

import pydotplus
import xml.etree.ElementTree as eTree
from copy import copy

"""
Gets a graphviz graph with coordinates and writes it to xml (.bpmn) format
Will rise an exception is data contains no coordinates


Example:

xml = XMLMaker()

# 1. Transform pydotplus.Dot() object to data and load it
pydot_graph = pydotplus.Dot()
graph_dot_data = pydot_graph.create(prog='dot', format='dot')  # создание данных в формате dot
xml.load_dot_data(graph_dot_data)

# 2. Or load a .gv file
xml.load_dot_file('bpmn.gv')

xml.make()
xml.write('bpmn.xml')
# or get a string representation
print(xml.to_string())
"""


class BPMNObject:
    """
    Abstract class for bpmn objects
    """

    def __init__(self, obj_type: str, obj_number: int):
        self.id = f'{obj_type}_{str(obj_number)}'

    def get_id(self) -> str:
        return self.id


class SequenceFlow(BPMNObject):
    type = 'sequenceFlow'
    counter = 0

    def __init__(self, source_node_id: str, dest_node_id: str):
        super().__init__(SequenceFlow.type, self.counter)
        self.sourceRef = source_node_id
        self.targetRef = dest_node_id
        self.pos = []  # [(x1, y1), (x2, y2),...]
        SequenceFlow.counter += 1

    def set_pos(self, pos: str):
        pos = pos[1:-1]  # remove '"'
        pos = pos.replace('e,', '')
        pairs = [pair for pair in pos.split(' ')]
        # the first pair (with 'e') is actually the destination point
        for pair in pairs[1:]:
            x_y = pair.split(',')
            # if there is a new line, '\' appears
            self.pos.append((float(x_y[0].replace("\\", '')), float(x_y[1].replace("\\", ''))))
        x_y = pairs[0].split(',')
        self.pos.append((float(x_y[0]), float(x_y[1])))

    def change_y_direction(self, lift: float):
        self.pos = [(t[0], -t[1] + lift) for t in self.pos]

    def get_source(self) -> str:
        return self.sourceRef

    def get_target(self) -> str:
        return self.targetRef

    def get_xy(self) -> list:
        return self.pos


class Node(BPMNObject):
    """
    Abstract class for bpmn shapes (excluding edges)
    """

    def __init__(self, obj_type: str, obj_number: int, position: str, height: str, width: str):
        super().__init__(obj_type, obj_number)
        self.width = float(width)
        self.height = float(height)
        self.x, self.y = (float(el) for el in position[1:-1].split(','))

    def inc_size(self, param: float):
        self.width *= param
        self.height *= param

    def move_center__down_left(self):
        """
        It is possible that the drawing program can interpret x and y not as a CENTER but as a LOW LEFT point
        of the figure; in this case we need to move the position of the figure to get a correct picture
        """
        self.x -= self.width / 2
        self.y -= self.height / 2

    def change_y_direction(self, lift: float):
        self.y = -self.y + lift

    def get_x(self) -> float:
        return self.x

    def get_y(self) -> float:
        return self.y

    def get_width(self) -> float:
        return self.width

    def get_height(self) -> float:
        return self.height


class Task(Node):
    type = 'task'
    counter = 0

    def __init__(self, name, position: str, height: str, width: str):
        super().__init__(Task.type, self.counter, position, height, width)
        # self.name = name.replace('"', '').replace("'", '')
        self.name = name
        self.incoming = set()
        self.outgoing = set()
        Task.counter += 1

    def set_incoming(self, incoming_edge: str):
        self.incoming.add(incoming_edge)

    def set_outgoing(self, outgoing_edge: str):
        self.outgoing.add(outgoing_edge)

    def get_incoming(self) -> set:
        return self.incoming

    def get_outgoing(self) -> set:
        return self.outgoing

    def get_name(self) -> str:
        return self.name


class StartEvent(Node):
    type = 'startevent'
    counter = 0

    def __init__(self, position: str, height: str, width: str):
        super().__init__(StartEvent.type, self.counter, position, height, width)
        self.outgoing = set()
        StartEvent.counter += 1

    def set_outgoing(self, outgoing_edge: str):
        self.outgoing.add(outgoing_edge)

    def get_outgoing(self):
        return self.outgoing


class EndEvent(Node):
    type = 'endevent'
    counter = 0

    def __init__(self, position: str, height: str, width: str):
        super().__init__(EndEvent.type, self.counter, position, height, width)
        self.incoming = set()
        EndEvent.counter += 1

    def set_incoming(self, incoming_edge: str):
        self.incoming.add(incoming_edge)

    def get_incoming(self) -> set:
        return self.incoming


class ParallelGateway(Node):
    type = 'parallelGateway'
    counter = 0

    def __init__(self, position: str, height: str, width: str):
        super().__init__(ParallelGateway.type, self.counter, position, height, width)
        self.incoming = set()
        self.outgoing = set()
        ParallelGateway.counter += 1

    def set_incoming(self, incoming_edge: str):
        self.incoming.add(incoming_edge)

    def set_outgoing(self, outgoing_edge: str):
        self.outgoing.add(outgoing_edge)

    def get_incoming(self) -> set:
        return self.incoming

    def get_outgoing(self) -> set:
        return self.outgoing


class ExclusiveGateway(Node):
    type = 'exclusiveGateway'
    counter = 0

    def __init__(self, position: str, height: str, width: str):
        super().__init__(ExclusiveGateway.type, self.counter, position, height, width)
        self.incoming = set()
        self.outgoing = set()
        ExclusiveGateway.counter += 1

    def set_incoming(self, incoming_edge: str):
        self.incoming.add(incoming_edge)

    def set_outgoing(self, outgoing_edge: str):
        self.outgoing.add(outgoing_edge)

    def get_incoming(self) -> set:
        return self.incoming

    def get_outgoing(self) -> set:
        return self.outgoing


class BPMN:
    """
    Creates and contains and modifies bpmn objects
    """

    def __init__(self, pydot_graph: pydotplus.Dot) -> None:
        self.bpmn_nodes, self.bpmn_edges = self._create_bpmn_objects(pydot_graph)
        self.bpmn_objects = self.bpmn_nodes + self.bpmn_edges

    def get_bpmn_nodes(self) -> list:
        return self.bpmn_nodes

    def get_bpmn_edges(self) -> list:
        return self.bpmn_edges

    def get_bpmn_objects(self) -> list:
        return self.bpmn_objects

    @staticmethod
    def _create_bpmn_objects(pydot_graph: pydotplus.Dot) -> (list, list):

        bpmn_nodes = []  # list of bpmn nodes
        nodes = {}  # nodes[node_name] = bpmn_node
        for graph_node in pydot_graph.get_node_list():
            name = graph_node.get_name()

            shape = graph_node.__get_attribute__('shape')
            label = graph_node.__get_attribute__('label')
            height = graph_node.__get_attribute__('height')
            width = graph_node.__get_attribute__('width')
            pos = graph_node.__get_attribute__('pos')
            if pos is not None:  # not to include 'system' nodes
                name = modify_str(name)
                label = modify_str(label)
                if name == 'startevent':
                    bpmn_node = StartEvent(pos, height, width)
                elif name == 'endevent':
                    bpmn_node = EndEvent(pos, height, width)
                elif shape == 'diamond':
                    bpmn_node = (ParallelGateway if (label == '+') else ExclusiveGateway)(pos, height, width)
                elif shape == 'box':
                    bpmn_node = Task(label, pos, height, width)
                else:
                    raise AssertionError(f'type of node {graph_node.get_name()} was not detected!')

                nodes[name] = bpmn_node
                bpmn_nodes.append(bpmn_node)
        bpmn_edges = []  # list of SequenceFlow() (bpmn edges)
        for graph_edge in pydot_graph.get_edge_list():
            source_name = modify_str(graph_edge.get_source())
            dest_name = modify_str(graph_edge.get_destination())
            source_node = nodes[source_name]
            dest_node = nodes[dest_name]

            pos = graph_edge.__get_attribute__('pos')

            bpmn_edge = SequenceFlow(source_node.get_id(), dest_node.get_id())
            bpmn_edge.set_pos(pos)
            source_node.set_outgoing(bpmn_edge.get_id())
            dest_node.set_incoming(bpmn_edge.get_id())
            bpmn_edges.append(bpmn_edge)

        return bpmn_nodes, bpmn_edges

    def change_graph_vertical_direction(self):
        """
        Rotates the graph - changes the vertical direction of the graph (down->up to up->down or vice-versa)
        """
        y_node_list = [bpmn_node.get_y() for bpmn_node in self.bpmn_nodes]
        min_y = min(y_node_list)
        max_y = max(y_node_list)
        for bpmn_edge in self.bpmn_edges:
            for x, y in bpmn_edge.get_xy():
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

        for bpmn_object in self.bpmn_objects:
            bpmn_object.change_y_direction(min_y + max_y)

    def inc_nodes_size(self, coeff: float = 72.72):
        """
        Transform size from inches to coordinates
        :param coeff: = 72.72 for correct transformation
        """
        for bpmn_node in self.bpmn_nodes:
            bpmn_node.inc_size(coeff)

    def move_nodes_centers__down_left(self):
        for bpmn_node in self.bpmn_nodes:
            bpmn_node.move_center__down_left()


class NoCoordinatesInDotData(Exception):

    def __init__(self) -> None:
        message = "DOT data does not contain coordinates. " \
                  "Creating a DOT data with coordinates must be done the following way: \n" \
                  "pydotplus.Dot().create(prog='dot', format='dot') - data; or \n" \
                  "pydotplus.Dot().write('<path/file_name>.gv', prog='dot', format='dot') - file."
        super().__init__(message)


class XMLMaker:
    """
    Main class: creates xml-tree from DOT-file
    """

    def __init__(self) -> None:
        self._pydot_graph = None
        self._root = None

    def load_dot_file(self, path):
        pydot_graph = pydotplus.graph_from_dot_file(path)

        coordinates_exist = self._check_for_coordinates(pydot_graph)
        if coordinates_exist:
            self._pydot_graph = pydot_graph
        else:
            raise NoCoordinatesInDotData()

    def load_dot_data(self, dot_data):
        pydot_graph = pydotplus.graph_from_dot_data(dot_data)

        coordinates_exist = self._check_for_coordinates(pydot_graph)
        if coordinates_exist:
            self._pydot_graph = pydot_graph
        else:
            raise NoCoordinatesInDotData()
        return self

    @staticmethod
    def _check_for_coordinates(pydot_graph):
        coordinates_exist = False
        for graph_node in pydot_graph.get_nodes():
            if graph_node.__get_attribute__('pos') is not None:
                coordinates_exist = True
                break
        return coordinates_exist

    def make(self, change_graph_vertical_direction: bool = True, move_nodes_centers__down_left: bool = True):
        """
        Main method
        :param change_graph_vertical_direction: changes the direction of the graph (from down->up to up->down or vice-versa)
        :param move_nodes_centers__down_left:
        :return:
        """
        self._root = self._create_definitions()
        bpmn = BPMN(self._pydot_graph)
        self._add_objects_and_connections(bpmn)

        # bpmn.inc_nodes_size must be before move_nodes_centers__down_left because size is used in moving node centers
        if change_graph_vertical_direction:
            bpmn.change_graph_vertical_direction()
        bpmn.inc_nodes_size()
        if move_nodes_centers__down_left:
            bpmn.move_nodes_centers__down_left()

        self._add_coordinates(bpmn)

    @staticmethod
    def _create_definitions():
        """
        Set the beginning of the xml-file
        """
        root = eTree.Element('bpmn:definitions')
        root.set("xmlns:bpmn", "http://www.omg.org/spec/BPMN/20100524/MODEL")
        root.set("xmlns:bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI")
        root.set("xmlns:dc", "http://www.omg.org/spec/DD/20100524/DC")
        root.set("xmlns:di", "http://www.omg.org/spec/DD/20100524/DI")
        root.set("id", "Definitions_123")
        root.set("targetNamespace", "http://bpmn.io/schema/bpmn")
        return root

    def _add_objects_and_connections(self, bpmn: BPMN):
        self.ProcessBuilder().build(bpmn, self._root)  # add objects and their connections to xml

    def _add_coordinates(self, bpmn):
        self.DiagramBuilder().build(bpmn, self._root)  # add coordinates of the objects to xml

    class ProcessBuilder:
        """
        Adds objects and their connections to xml
        """

        def __init__(self) -> None:
            self.prefix = 'bpmn'

        def pref(self, s: str) -> str:
            return f'{self.prefix}:{s}'

        @staticmethod
        def get_process_id():
            return "Process_123"

        def add_incoming(self, bpmn_element, bpmn_object):
            for incoming in bpmn_object.get_incoming():
                bpmn_element.set(self.pref("incoming"), incoming)

        def add_outgoing(self, bpmn_element, bpmn_object):
            for outgoing in bpmn_object.get_outgoing():
                bpmn_element.set(self.pref("outgoing"), outgoing)

        def build(self, bpmn: BPMN, parent: eTree.SubElement):
            process = eTree.SubElement(parent, self.pref('process'))
            process.set("id", self.get_process_id())

            for bpmn_object in bpmn.get_bpmn_objects():
                if isinstance(bpmn_object, StartEvent):
                    bpmn_element = eTree.SubElement(process, self.pref('startEvent'))
                    self.add_outgoing(bpmn_element, bpmn_object)
                elif isinstance(bpmn_object, EndEvent):
                    bpmn_element = eTree.SubElement(process, self.pref('endEvent'))
                    self.add_incoming(bpmn_element, bpmn_object)
                elif isinstance(bpmn_object, Task):
                    bpmn_element = eTree.SubElement(process, self.pref('task'))
                    bpmn_element.set("name", bpmn_object.get_name())
                    self.add_incoming(bpmn_element, bpmn_object)
                    self.add_outgoing(bpmn_element, bpmn_object)
                elif isinstance(bpmn_object, (ParallelGateway, ExclusiveGateway)):
                    obj_type = 'parallelGateway' if isinstance(bpmn_object, ParallelGateway) else 'exclusiveGateway'
                    bpmn_element = eTree.SubElement(process, self.pref(obj_type))
                    self.add_incoming(bpmn_element, bpmn_object)
                    self.add_outgoing(bpmn_element, bpmn_object)
                elif isinstance(bpmn_object, SequenceFlow):
                    bpmn_element = eTree.SubElement(process, self.pref('sequenceFlow'))
                    bpmn_element.set("sourceRef", bpmn_object.get_source())
                    bpmn_element.set("targetRef", bpmn_object.get_target())
                else:
                    raise AssertionError('type {} was not processed'.format(type(bpmn_object)))

                bpmn_element.set("id", bpmn_object.get_id())

    class DiagramBuilder:
        """
        Adds coordinates of the objects to xml
        """

        def __init__(self) -> None:
            self.prefix = 'bpmndi'
            self.shape_prefix = 'dc'
            self.edge_prefix = 'di'

        def pref(self, s: str) -> str:
            return f'{self.prefix}:{s}'

        def s_pref(self, s: str) -> str:
            """
            Return the prefix for subelements of Shape
            """
            return f'{self.shape_prefix}:{s}'

        def e_pref(self, s: str) -> str:
            """
            Return the prefix for subelements of Edge
            """
            return f'{self.edge_prefix}:{s}'

        @staticmethod
        def modify_id(s: str) -> str:
            return f'{s}_element'

        def build(self, bpmn: BPMN, parent: eTree.SubElement):
            diagram = eTree.SubElement(parent, self.pref('BPMNDiagram'))
            diagram.set("id", "Diagram_123456")
            plane = eTree.SubElement(diagram, self.pref('BPMNPlane'))
            plane.set("id", "Plane_123456")
            plane.set("bpmnElement", XMLMaker.ProcessBuilder.get_process_id())

            for bpmn_node in bpmn.get_bpmn_nodes():
                bpmn_element = eTree.SubElement(plane, self.pref('BPMNShape'))
                bpmn_element.set("id", self.modify_id(bpmn_node.get_id()))
                bpmn_element.set("bpmnElement", bpmn_node.get_id())
                bounds = eTree.SubElement(bpmn_element, self.s_pref("Bounds"))
                bounds.set("x", str(round(bpmn_node.get_x(), 3)))
                bounds.set("y", str(round(bpmn_node.get_y(), 3)))
                bounds.set("width", str(bpmn_node.get_width()))
                bounds.set("height", str(bpmn_node.get_height()))

            for bpmn_edge in bpmn.get_bpmn_edges():
                bpmn_element = eTree.SubElement(plane, self.pref('BPMNEdge'))
                bpmn_element.set("id", self.modify_id(bpmn_edge.get_id()))
                bpmn_element.set("bpmnElement", bpmn_edge.get_id())
                for x, y in bpmn_edge.get_xy():
                    waypoint = eTree.SubElement(bpmn_element, self.e_pref('waypoint'))
                    waypoint.set("x", str(round(x, 3)))
                    waypoint.set("y", str(round(y, 3)))

    def write(self, path):
        with open(path, mode='w', encoding='utf-8') as f:
            f.write(self.to_string())

    def to_string(self, pretty_print: bool = True):
        root = copy(self._root)
        if pretty_print:
            root = self._pretty_print(root)  # pretty print of xml

        tree = eTree.ElementTree(root).getroot()
        return eTree.tostring(tree, encoding='utf-8').decode('utf-8')

    def _pretty_print(self, elem: eTree.Element, level: int = 0):
        """
        Helper function, adds indentation to XML output. ('pretty print')
        :param elem: object of Element class, representing element to which method adds indentation,
        :param level: current level of indentation.
        """
        i = "\n" + level * "\t"
        j = "\n" + (level - 1) * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for subelem in elem:
                self._pretty_print(subelem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = j
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = j
        return elem


def modify_str(s):
    return s[1:-2]
