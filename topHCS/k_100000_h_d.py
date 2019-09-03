import math
import numpy as np
import copy
import torch
from csvec import CSVec
from topHCS import TopHCS
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import ipdb
import sys

fig, axs = plt.subplots(2, 2, figsize=(20, 18))
matplotlib.rcParams.update({'font.size': 16})

device = 'cuda' 
graphPath = 'graphs/k_100000_h_d.png'

initialVecPaths = ["../../datafiles/initialVs0.torch"] 
accumVecPaths = ["../../datafiles/accumulatedVs0.torch"] 

#stepsize = 1000
for p, paths in enumerate([initialVecPaths, accumVecPaths]):
	print("Using {}".format(paths))
	vecs = [torch.load(paths[0], map_location=device)]
	print("vector loaded")
	summed = vecs[0].clone()
	d = len(summed)

	expectedIndices = torch.sort(summed**2)[1]

	kVals = [50000, 60000, 70000, 80000, 90000, 100000]
	hVals = [d]
	cVals = [180000]
	for c, cols in enumerate(cVals):
		csvecAcc = np.zeros(len(kVals))
		topHCSAcc = np.zeros((len(hVals), len(kVals)))
		csvecL2 = np.zeros(len(kVals))
		topHCSL2 = np.zeros((len(hVals), len(kVals)))
		for k_i, k in enumerate(kVals): 
			d, c, r, numBlocks = len(summed), cols, 15, 30
			#ipdb.set_trace()
			expected = torch.zeros(len(summed), device=device)
			expected[expectedIndices[-k:].to(device)] = summed[expectedIndices[-k:].to(device)]
			
			assert(summed.size() == vecs[0].size())
			w_0 = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w_0 += vecs[0]
			print("worker added")
			
			result = w_0.unSketch(k)
			csvecAcc[k_i] = (expected[expectedIndices] * result[expectedIndices]).nonzero().numel() / k
			print("k = {}".format(k))
			csvecL2[k_i] = (torch.sum((result - expected)**2))**0.5
			for h_i, h in enumerate(hVals):
				result = torch.zeros(len(summed), device=device)
				w_0 = TopHCS(d=d, c=c, r=r, h=h, numBlocks=numBlocks, device=device)
				w_0.store(vecs[0])
				print("worker added")
				workers = [w_0]
				result = TopHCS.topKSum(workers, k)
				 
				topHCSAcc[h_i, k_i]  = (expected[expectedIndices] * result[expectedIndices]).nonzero().numel() / k
				print("h_curr = {}".format(h_curr))
				topHCSL2[h_i, k_i] = (torch.sum((result - expected)**2))**0.5
			print('topHCS accuracy :', '\n', topHCSAcc)
			print('CSVec accuracy :', '\n', csvecAcc)
		numColors = len(hVals) + 1
		colors = plt.cm.plasma(np.linspace(0,1,numColors))
		axs[p, 0].set_xlabel("k")
		axs[p, 0].set_ylabel("index accuracy")
		axs[p, 0].set_title("index accuracy vs. k, sketch=({}, {})".format(r, c))
		axs[p, 0].plot(kVals, csvecAcc, marker='x', color=colors[0], label="CSVec (h=0)")
		axs[p, 1].set_xlabel("k")
		axs[p, 1].set_ylabel("L2 reconstruction error")
		axs[p, 1].set_title("L2 reconstruction error vs. k")
		axs[p, 1].plot(kVals, csvecL2, marker='x', color=colors[0], label="CSVec (h=0)")
		for h_i, h in enumerate(hVals):
			axs[p, 0].plot(kVals, topHCSAcc[h_i], marker='x', color=colors[h_i+1], label="h = k*{}".format(h))
			axs[p, 1].plot(kVals, topHCSL2[h_i], marker='x', color=colors[h_i+1], label="h = k*{}".format(h))
		axs[p, 0].legend()
		axs[p, 1].legend()
plt.savefig(graphPath)
