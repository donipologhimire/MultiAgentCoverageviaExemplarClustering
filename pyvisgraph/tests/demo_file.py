import sys
sys.path.append("C:\\Users\\donip\\Desktop\\UCI\\UCI_Research\\YEAR_4\\non_convex_coverage_files\\non_convex_coverage\\pyvisgraph\\")

import graph 
from pyvisgraph.visgraph import VisGraph


## 1. Generate a random polygon

polys = [
    [Point(2.0,1.0), Point(3.5,4.0), Point(5.0,1.0), Point(4.0,2.0)],
    [Point(6.0,4.0), Point(9.0,4.0), Point(7.5,8.0)],
    [Point(1.0,6.0), Point(2.0,6.0), Point(2.0,10.0), Point(3.0,10.0), Point(3.0,5.0), Point(1.0,5.0)] # Non-convex polygon
]

g = VisGraph()
g.build(polys)