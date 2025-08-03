class Point:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z

class Curve:
    def __init__(self, id, curve_type, start_point=None, end_point=None, center_point=None, radius=None, **kwargs):
        self.id = id
        self.type = curve_type
        self.start_point = start_point
        self.end_point = end_point
        self.center_point = center_point
        self.radius = radius
        self.extra = kwargs           

class Constraint:
    def __init__(self, id, constraint_type, entities):
        self.id = id
        self.type = constraint_type
        self.entities = entities  

class Sketch:
    def __init__(self, id, name, sketch_type):
        self.id = id
        self.name = name
        self.type = sketch_type
        self.points = {}      
        self.curves = {}      
        self.constraints = {}  
