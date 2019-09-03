import math
import numpy as np
import copy
import torch
from csvec import CSVec

LARGEPRIME = 2**61-1

cache = {}
def topk(vec, k):
	""" Return the largest k elements (by magnitude) of vec"""
	ret = torch.zeros_like(vec)
	if k != 0:
		topkIndices = torch.sort(vec**2)[1][-k:]
		ret[topkIndices] = vec[topkIndices]
	return ret

class TopHCS(object): # represents a worker

    def __init__(self, d, c, r, h, numBlocks, device='cpu'): 
        self.h = h
        self.device = device
        self.topH = torch.zeros(d, dtype=torch.float, device=self.device)
        # temporarily remove self.bottomH to save memory
	# self.bottomH = torch.zeros(d, dtype=torch.float, device=self.device)
        self.csvec = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=self.device)
        
    def store(self, vec):
        assert(self.topH.nonzero().numel() == 0)
        self.csvec.zero()
        self.topH = topk(vec, self.h).to(self.device)
        #self.bottomH = (vec - self.topH).to(self.device)
        self.csvec += (vec - self.topH).to(self.device)
        
    @classmethod
    def topKSum(cls, workers, k):
        sketchSum = workers[0].csvec 
        topHSum = workers[0].topH
        for w in workers[1:]:
            sketchSum += w.csvec
            topHSum += w.topH
        d = len(topHSum)
        unSketchedSum = sketchSum.unSketch(k=d)
        ret = topk(topHSum + unSketchedSum, k)
        return ret
