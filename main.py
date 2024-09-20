#c= {[a,b]:["element",property]}
# c = {[0,1]:['VDC',10],[1,2]:'resistor',[2,3]:'wire',[3,0]:'wire'}

class element:
    def __init__(self,elementType:str,connectedNodes:list,property: int|float):
        self.elementType = elementType
        self.connectedNodes = connectedNodes
        self.property = property



    def __str__(self):
        return f"{self.elementType}"


class node:
    
    #properties --> 
    def __init__(self, nodeID:int,nodeVolt:int|float):
        self.nodeID = nodeID
        self.nodeVolt = nodeVolt

    def __str__(self):
        return f"{self.node}"


class circuit:
    def __init__(self, nodes:node, elements:element):
        self.nodes = nodes
        self.elements = elements