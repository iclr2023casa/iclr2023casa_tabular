import numpy as np
from math import floor

class GridWorld:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.now = 0
        self.aval_a = [(0,-1),(0,1),(-1,0),(1,0)]
    
    def reset(self):
        self.now = [0, 0]
        return self.now[0]*self.num_rows + self.now[1]
    
    def step(self, a):
        aval_a = self.aval_a[a]
        new = [self.now[0]+aval_a[0],self.now[1]+aval_a[1]]
        new[0] = min(self.num_cols-1,max(0,new[0]))
        new[1] = min(self.num_rows-1,max(0,new[1]))
        self.now=new
        if new[0]==self.num_cols-1 and new[1]==self.num_rows-1:
            d = True
            r = 1
        else:
            d= False
            r= - 0.1
        return self.now[0]*self.num_rows + self.now[1], r,d,{}
        