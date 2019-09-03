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

device = 'cpu' 
vecs = []
for i in range(4):
	sampleVec = torch.randint(-10, 10, (40,), dtype=torch.float)
	vecs.append(sampleVec)
assert(len(vecs) == 4)

for q, paths in enumerate([1]):
	
	summed = vecs[0].clone()
	for v in vecs[1:]:
		summed += v	

	expectedIndices = torch.sort(summed**2)[1]

	kVals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	hVals = [0, 1.0]
	cVals = [50, 100, 1000, 10000]
	cVals = [20]
	for p, cols in enumerate(cVals):
		csvec_accuracy = np.zeros(len(kVals))
		topHCS_accuracy = np.zeros((len(hVals), len(kVals)))
		for k_i, k in enumerate(kVals): 
			d, c, r, numBlocks = len(summed), cols, 15, 1
			#ipdb.set_trace()
			expected = torch.zeros(len(summed), device=device)
			expected[expectedIndices[-k:].to(device)] = summed[expectedIndices[-k:].to(device)]
			
			assert(summed.size() == vecs[0].size())
			w_0 = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w_0 += vecs[0]
			print("")
			w_1 = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w_1 += vecs[1]

			print("")
			w_2 = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w_2 += vecs[2]

			print("")
			w_3 = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w_3 += vecs[3]

			print("")
			workers_summed = w_0 + w_1 + w_2 + w_3
			result = workers_summed.unSketch(k)

			#csvec_accuracy = (expected == result).nonzero().numel() / k
			csvec_accuracy[k_i] = (expected[expectedIndices] * result[expectedIndices]).nonzero().numel() / k
			print("k = ", k)
			#print("num of non 0 elements in expected : ", expected[expectedIndices].nonzero().numel())
			#print("num of non 0 elements in result : ", result[expectedIndices].nonzero().numel())
			print('csvec accuracy :', csvec_accuracy)
			print('expected :', expected)
			print('result :', result)
			
			for h_i, h in enumerate(hVals):
				result = torch.zeros(len(summed), device=device)
				h_curr = int(h*k)
				w_0 = TopHCS(d=d, c=c, r=r, h=h_curr, numBlocks=numBlocks, device=device)
				w_0.store(vecs[0])
				print(w_0.topH)

				print("")
				w_1 = TopHCS(d=d, c=c, r=r, h=h_curr, numBlocks=numBlocks, device=device)
				w_1.store(vecs[1])

				print(w_0.topH)
				print("")
				w_2 = TopHCS(d=d, c=c, r=r, h=h_curr, numBlocks=numBlocks, device=device)
				w_2.store(vecs[2])

				print(w_0.topH)
				print("")
				w_3 = TopHCS(d=d, c=c, r=r, h=h_curr, numBlocks=numBlocks, device=device)
				w_3.store(vecs[3])

				print(w_0.topH)
				print("")
				workers = [w_0, w_1, w_2, w_3]
				result = TopHCS.topKSum(workers, k)
				 
				#topHCS_accuracy = (expected == result).nonzero().numel() / k
				#print("num of non 0 elements in topH result : ", result[expectedIndices].nonzero().numel())
				topHCS_accuracy[h_i, k_i]  = (expected[expectedIndices] * result[expectedIndices]).nonzero().numel() / k
				print('result for h =', h_curr, " :", result)
			print('topHCS accuracy :', '\n', topHCS_accuracy)
