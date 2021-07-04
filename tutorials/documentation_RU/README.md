#Краткое описание основных классов и модулей библиотеки SberProcessMining (SberPM)


#sberpm
##sberpm\_holder.py


    class DataHolder
Загружаемый объект обрабатывается. Содержит журнал событий и имена его столбцов.

- Методы:
    
  `def _preprocess_data(self, df, time_format, time_errors, dayfirst, yearfirst):`
    Выполняет базовую предварительную обработку:
    - удаляет нулевые значения;
    - преобразует основные столбцы в тип str
    - преобразует метки времени в формат date_time (если указан столбец dt_column);
    - сортирует по идентификатору и метке времени (если заданы столбец start_timestamp_column или столбец end_timestamp_column).
    
    `def get_grouped_data(self, *columns):`
    Возвращает данные, сгруппированные по идентификатору, с заданными столбцами, агрегированными в кортежи.
    
    `def get_grouped_columns(self, *columns):`
    Возвращает отдельные столбцы (серии) данных, агрегированных по заданным столбцам.
    
    `def get_unique_activities(self):`
    Возвращает уникальные действия в журнале событий.
    
    `def get_columns(self):`
    Возвращает столбцы данных.
    
    `def get_text(self):`
    Возвращает текстовый столбец данных или Нет, если столбец text_column не был задан в конструкторе.
    
    `def _groupby(data, groupby_column, *agg_columns):`
    Группирует данные по идентификатору и объединяет все заданные столбцы в кортежи.
    
    `def check_or_calc_duration(self):`
    Рассчитывает продолжительность, если она не рассчитана.


##sberpm\_utils.py
- Функции:

  `def generate_data_partitions(df, id_column, batch_num):`
    Работа с индексами для трассировок событий.


##sberpm\autoinsights
###sberpm\autoinsights\_auto_insights.py


    class AutoInsights
Автоматический алгоритм поиска информации.

- Методы:
    
  `def apply(self,miner,mode='overall',width_by_insight=True, q_min=0.1, q_top=0.85):`
    Вычисляет информацию.
    
  `def set_success_activities(self, success_activities):`
    Вычисляет информацию.
    
  `def set_failure_activities(self, failure_activities):`
    Устанавливает действия при ошибке.
    
  `def _calculate_edge_labels(self, edge_stats, mode, edge_name):`
    Устанавливает метки для ребер.
    
  `def _get_color(insights, mode, name_column):`
    Задает цвета для объектов (узлов или ребер) в соответствии со статусом "insight".
    
  `def _add_legend(self, stats, mode):`
    Добавляет два информационных узла к графику.


##sberpm\bpmn
###sberpm\bpmn\_bpmn_file_to_graph.py


    class LocalObject
Абстрактный класс. Представляет объект, импортированный из файла bpmn.


    class LocalNode(LocalObject)
Представляет узел, импортированный из файла bpmn.


- Методы:
    
  `def get_subgraph_objects(self):`
    Возвращает внутренние объекты, которые не являются отдельными узлами.
    
  `def get_start_nodes(self):`
    Возвращает объект "startEvent" этого объекта и его внутренних объектов.
    
  `def get_end_nodes(self):`
    Возвращает объект "endEvent" этого объекта и его внутренних объектов.


    class LocalEdge(LocalObject)
Представляет ребро, импортированное из файла bpmn.

    class DataContainer
Представляет собой график. Содержит информацию об узлах и ребрах.

- Методы:

  `def add_edge(self, edge):`
    Добавляет объект LocalEdge() в локальные структуры данных.
    
  `def remove_edge(self, in_node, out_node):`
    Удаляет объект LocalEdge() из локальных структур данных.
    
  `def remove_node(self, node):`
    Удаляет объект LocalNode() из локальных структур данных.
   
 
    class BpmnImporte
Загружает файл .xml/.bpmn, содержащий график в формате bpmn, и визуализирует его.

- Методы:

  `def load_bpmn_from_xml(self, file_path, additional_tags_to_ignore: (list, None) = None, remove_gateways=False):`
    Считывает XML-файл из заданного пути к файлу и отображает его во внутреннее представление диаграммы BPMN.
    
  `def _read_xml_file(file_path):`
    Считывает XML-файл BPMN 2.0.
    
  `def _import_nodes_and_dependant_nodes(self, element: minidom.Element, parent_node: (LocalNode, None)):`
    Импортирует элемент объекта из bpmn (создает объект локального узла).
    Если предполагается, что у этого объекта есть дочерние элементы ("process" или "subProcess"), эта функция вызывается рекурсивно.
    
  `def _import_edge(self, edge_element: minidom.Element):`
    Импортирует элемент потока из bpmn (создает LocalEdge объект).
    
  `def _make_call_elements_children(self):`
    Если тип объекта "callActivity" и у него есть ссылка "calledElement",
    этот объект становится родительским узлом для узла "calledElement".
    
  `def _make_edges_for_references(self):`
    Если узел имеет атрибут 'attachedToRef', создает ребро
    от узла 'attachedToRef' к этому узлу.
    
  `def _remove_gateways(self):`
    Удаляет шлюзы из графика.
    
  `def _ensure_all_connections_are_between_separate_nodes(self):`
    Если исходным объектом ребра является узел, у которого есть дочерние элементы,
    мы предполагаем, что этот узел не должен отображаться на графике.
    Соединение между этим узлом и целевым узлом заменяется
    с подключениями от его внутренних узлов (children) и целевого узла.
    
  `def _get_attributes(element: minidom.Element):`
    Возвращает атрибуты элемента.
    
  `def _remove_namespace_from_tag_name(tag_name: str):`
    Удаляет аннотацию пространства имен из имени тега (например, semantic:startEvent -> startEvent).
    
  `def _iterate_elements(element: minidom.Element):`
    Выполняет итерацию по дочерним Nodes/Elements родительского Node/Element.
    
  `def get_pydotplus_graph(self, show_edge_labels=False, vertical=True):`
    Возвращает объект визуализации импортированного bpmn-graph.


    class PydotPlusGraphMaker
Класс использует пакет pydotplus (внутри которого есть graphviz) для визуализации импортированного графика bpmn.

- Методы:
  
  `def make_graph(data, show_edge_labels=True, vertical=True, orthogonal_lines=False):`
    Возвращает объект визуализации импортированного bpmn-graph.
  
  `def _create_pydot_node(node, graph, pydot_node_id_dict):`
    Создает pydot.Node() и добавляет его в родительский график.
  
  `def _modify_id(string):`
    Изменяет строку (id, name,...), поскольку graphviz может неправильно работать с определенными символами.


###sberpm\bpmn\_bpmn_graph_to_file
####sberpm\bpmn\_bpmn_graph_to_file\_bpmn_exporter.py


    class BpmnExporter
Преобразует сеть Петри в график BPMN и сохраняет его в файл .bpmn.

- Методы:

  `def apply_petri(self, petri_net):`
    Преобразует заданную сеть Петри в график BPMN.
  
  `def write(self, filename):`
    Сохраняет рассчитанный график BPMN в нотации BPMN в файл.
  
  `def get_string_representation(self):`
    Возвращает строковое представление записи BPMN вычисленного графика BPMN.


####sberpm\bpmn\_bpmn_graph_to_file\_bpmn_to_dot.py
- Функции:
  
  `def bpmn_to_graph(bpmn_graph):`
    Преобразует данный график bpmn в объект графика, который можно визуализировать.
  
  
####sberpm\bpmn\_bpmn_graph_to_file\_bpmn_xml_maker.py
Получает график graphviz с координатами и записывает его в формат xml (.bpmn)
Вызовет исключение, если данные не содержат координат

    class BPMNObject
Абстрактный класс для объектов bpmn

    class Node(BPMNObject)
Абстрактный класс для фигур bpmn (за исключением ребер)


- Методы:
  
    `def move_center__down_left(self)`
  Возможно, что программа рисования может интерпретировать x и y не как CENTER, а как LOW LEFT точку
фигуры; в этом случае нам нужно переместить положение фигуры, чтобы получить правильное изображение
  



```
Документация в процессе доработки. 
Многие методы не имеют комментариев и требуют отдельного изучения
```