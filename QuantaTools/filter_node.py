from .quanta_filter import QuantaFilter
from .quanta_type import QuantaType 

from .useful_node import position_name, position_name_to_int, row_location_name, location_name, NodeLocation, UsefulNode  
from .useful_info import UsefulInfo, useful_info


class FilterNode:
    def evaluate(self, useful_node):
        pass

    def describe(self):
        pass

                                       
class FilterAnd(FilterNode):
    def __init__(self, *args):
        self.children = args

    def evaluate(self, test_node):
        return all(child.evaluate(test_node) for child in self.children)

    def describe(self):
        answer = " and( "
        for child in self.children:
          answer += child.describe() + ", "
        answer += ")"
        return answer


class FilterOr(FilterNode):
    def __init__(self, *args):
        self.children = args

    def evaluate(self, test_node):
        return any(child.evaluate(test_node) for child in self.children)

    def describe(self):
        answer = " or( "
        for child in self.children:
          answer += child.describe() + ", "
        answer += ")"
        return answer    


class FilterHead(FilterNode):
    def __init__(self):
      pass

    def evaluate(self, test_node):
        return test_node.is_head

    def describe(self):
         return "IsHead" 

                                       
class FilterNeuron(FilterNode):
    def __init__(self):
      pass

    def evaluate(self, test_node):
        return not test_node.is_head

    def describe(self):
         return "IsNeuron" 

                                       
class FilterPosition(FilterNode):
    def __init__(self, position_name, filter_strength = QuantaFilter.MUST):
        self.filter_strength = filter_strength
        self.position = position_name_to_int(position_name)

    def evaluate(self, test_node):
        if self.filter_strength in [QuantaFilter.MUST, QuantaFilter.CONTAINS]:
          return (test_node.position == self.position)
        if self.filter_strength == QuantaFilter.NOT:
          return (not test_node.position == self.position)
        if self.filter_strength == QuantaFilter.MAY:
          return True 
        if self.filter_strength == QuantaFilter.MUST_BY:
          return (test_node.position <= self.position)      
        return False

    def describe(self):
        return self.filter_strength.value + " " + position_name(self.position)

                                       
class FilterContains(FilterNode):
    def __init__(self, quanta_type, minor_tag, filter_strength = QuantaFilter.MUST):
        self.quanta_type = quanta_type
        self.minor_tag = minor_tag
        self.filter_strength = filter_strength

    def evaluate(self, test_node):
        if self.filter_strength in [QuantaFilter.MUST, QuantaFilter.CONTAINS]:
          return test_node.contains_tag(self.quanta_type, self.minor_tag)   
        if self.filter_strength == QuantaFilter.NOT:
          return not test_node.contains_tag(self.quanta_type, self.minor_tag)
        if self.filter_strength == QuantaFilter.MAY:
          return True  
        return False

    def describe(self):
        return self.filter_strength.value + " " + self.quanta_type + " " + self.minor_tag


class FilterAttention(FilterContains):
    def __init__(self, minor_tag, filter_strength = QuantaFilter.MUST):
      super().__init__(QuantaType.ATTENTION, minor_tag, filter_strength)



class FilterImpact(FilterContains):
    def __init__(self, minor_tag, filter_strength = QuantaFilter.MUST):
      super().__init__(QuantaType.IMPACT, minor_tag, filter_strength)


class FilterPCA(FilterContains):
    def __init__(self, minor_tag, filter_strength = QuantaFilter.MUST):
      super().__init__(QuantaType.PCA, minor_tag, filter_strength)


class FilterAlgo(FilterContains):
    def __init__(self, minor_tag, filter_strength = QuantaFilter.MUST):
      super().__init__(QuantaType.ALGO, minor_tag, filter_strength)



def filter_nodes( the_nodes, the_filters):
  answer = []

  for test_node in the_nodes:
    if the_filters.evaluate(test_node):
      answer += [test_node]

  return answer