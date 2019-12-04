import math
import numpy as np
import copy
import torch
from csvec import CSVec

LARGEPRIME = 2**61-1

def topk(vec, k):
	""" Return the largest k elements (by magnitude) of vec """
	ret = torch.zeros_like(vec)
	if k != 0:
		topkIndices = torch.sort(vec**2)[1][-k:]
		ret[topkIndices] = vec[topkIndices]
	return ret

class TopHCS(object):
	""" Represents one worker"""
	def __init__(self, d, c, r, h, numBlocks, device='cpu'): 
		self.h, self.d = h, d 
		self.device = device
		self.topH = torch.zeros(d, dtype=torch.float, device=self.device)
		self.csvec = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=self.device)
	
	def zero(self):
		""" Clear csvec and topH tensor """
		self.csvec.zero()
		self.topH = torch.zeros(self.d, dtype=torch.float, device=self.device)
        
        # formerly store(...)
	def accumulateVec(self, vec):
		""" Compresses vector """
		""" Save top-h elements in self.topH, sketch bottom d-h elements """
		""" csvec and topH should be zero before storing """
		# assert(self.topH.nonzero().numel() == 0)
                # changed this for commefficient optimizer
		self.topH = topk(vec, self.h).to(self.device)
		self.csvec.accumulateVec((vec - self.topH).to(self.device))
        
	def accumulateTable(self, table):
		if table.size() != self.csvec.table.size():
			msg = "Passed in table has size {}, expecting {}"
			raise ValueError(msg.format(table.size(), self.csvec.table.size()))
		else:
			self.csvec.accumulateTable(table)

	@classmethod
	def topKSum(cls, workers, k, unSketchNum=0):
		assert isinstance(workers, list), "workers must be a list"
		sketchSum = copy.deepcopy(workers[0].csvec) 
		sketchSum.zero()
		topHSum = torch.zeros_like(workers[0].topH)
		for w in workers:
			sketchSum.accumulateTable(w.csvec.table)
			topHSum += w.topH
		d = len(topHSum)
		unSketchNum = d if (unSketchNum == 0) else unSketchNum
		unSketchedSum = sketchSum.unSketch(k=unSketchNum) 
		if topHSum.size() != unSketchedSum.size():
			msg = "topHSum has size {}, unSketchedSum size {}"
			raise ValueError(msg.format(topHSum.size(), unSketchedSum.size()))
		ret = topk(topHSum + unSketchedSum, k)
		return ret

