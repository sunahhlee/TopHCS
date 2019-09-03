import math
import numpy as np
import copy
import torch
from csvec import CSVec
from topHCS import TopHCS
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime

startTime = datetime.now()
device = 'cuda' 

#graphing
fig, axs = plt.subplots(2, 3, figsize=(24, 12))

#Loading vectors from file
initialVecPaths = ["../../datafiles/initialVs0.torch", 
                   "../../datafiles/initialVs1.torch", 
                   "../../datafiles/initialVs2.torch", 
                   "../../datafiles/initialVs3.torch"]
accumVecPaths = ["../../datafiles/accumulatedVs0.torch", 
                   "../../datafiles/accumulatedVs1.torch", 
                   "../../datafiles/accumulatedVs2.torch", 
                   "../../datafiles/accumulatedVs3.torch"]

for p, paths in enumerate([initialVecPaths, accumVecPaths]):
	vecs = []
	for path in paths:
		vec = torch.load(path, map_location=device)
		vecs.append(vec)

	#TODO: use torch.stack. I got lazy here
	summed = vecs[0]
	for lazy in vecs[1:]:
		summed += lazy 
	assert(vecs[0].size() == summed.size())

	d, c, r, numBlocks = len(summed), 540000, 5, 30
	kVals = [50000, 100000, 150000, 200000, 250000, 500000]
	hVals = [1.00, .75, .50, .25]

	# init CSVec stuff
	indexAcc_1 = np.zeros(len(kVals))
	L1_1 = np.zeros(len(kVals))
	L2_1 = np.zeros(len(kVals))

	# init TopHCS stuff
	indexAcc_2 = np.zeros((len(hVals), len(kVals))) 
	L1_2 = np.zeros((len(hVals), len(kVals))) 
	L2_2 = np.zeros((len(hVals), len(kVals))) 

	for k_i, k in enumerate(kVals):
		k = int(k) #stupid
		expected = torch.zeros(d, device=device)
		expectedIndices = torch.sort(summed**2)[1][-k:]
		expected[expectedIndices.to(device)] = summed[expectedIndices.to(device)]

		# CSVecs
		workers = []
		for vec in vecs:	
			w = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w += vec
			workers.append(w)

		# Summing CSVecs into 1st worker to save memory
		for w in workers[1:]:
			workers[0] += w 
		recovered_1 = workers[0].unSketch(k) 
		indexAcc_1[k_i] = (expected[expectedIndices] * recovered_1[expectedIndices]).nonzero().numel() / k 
		
		diff = recovered_1[recovered_1 != 0] - expected[recovered_1 != 0]
		L1_1[k_i] = torch.median(torch.abs(diff))
		L2_1[k_i] = torch.median(diff**2)

		for h_i, hVal in enumerate(hVals):
			h = int(k * hVal)
			workers = [] #reusing 
			for vec in vecs:	
				w = TopHCS(h=h, d=d, c=c, r=r, numBlocks=numBlocks, device=device)
				w.store(vec)
				workers.append(w)

			assert(len(workers) == len(vecs))

			recovered_2 = TopHCS.topKSum(workers, k) 
			indexAcc_2[h_i, k_i] = (expected[expectedIndices] * recovered_2[expectedIndices]).nonzero().numel() / k 
		
			diff = recovered_2[recovered_2 != 0] - expected[recovered_2 != 0]
			L1_2[h_i, k_i] = torch.median(torch.abs(diff))
			L2_2[h_i, k_i] = torch.median(diff**2)

			#indexAcc = (expected == recovered).nonzero().numel() / k
			# Use above only when k == d
			#print("\n", "k = %r; index accuracy = %r" % (k, indexAcc))

	colors=["#A30CE8", "#FF0000", "#E8710C", "#FFD20D"]

	axs[p, 0].plot(kVals, L1_1, color='blue', label="CSVec") 
	axs[p, 1].plot(kVals, L2_1, color='blue', label="CSVec") 
	axs[p, 2].plot(kVals, indexAcc_1, color='blue', label="CSVec") 
	for h_i, h in enumerate(hVals):
		axs[p, 0].plot(kVals, L1_2[h_i], color=colors[h_i], label="h = k*"+str(h)) 
		axs[p, 1].plot(kVals, L2_2[h_i], color=colors[h_i], label="h = k*"+str(h)) 
		axs[p, 2].plot(kVals, indexAcc_2[h_i], color=colors[h_i], label="h = k*"+str(h))
	axs[p, 0].set_xlabel("k for topK")
	axs[p, 1].set_xlabel("k for topK")
	axs[p, 2].set_xlabel("k for topK")
	axs[p, 0].set_ylabel("Median L1 Reconstruction Error")
	axs[p, 1].set_ylabel("Median L2 Reconstruction Error")
	axs[p, 2].set_ylabel("Index Accuracy Rate")
	axs[p, 0].set_title("Median L1 Reconstruction Error vs k")
	axs[p, 1].set_title("Median L2 Reconstruction Error vs k")
	axs[p, 2].set_title("Index Accuracy Rate vs k")
	
	axs[p, 0].legend()
	axs[p, 1].legend()
	axs[p, 2].legend()

	print("On %r" % (paths))
	print("k Values = %r" % (kVals))
	print("\n", "CSVec: index accuracy = %r" % (indexAcc_1))
	print("CSVec: median L1 reconstruction error = %r" % (L1_1))
	print("CSVec: median L2 reconstruction error = %r" % (L2_2))
	print("\n", "TopHCS: index accuracy = %r" % (indexAcc_2))
	print("TopHCS: median L1 reconstruction error = %r" % (L1_2))
	print("TopHCS: median L2 reconstruction error = %r" % (L2_2))
plt.savefig("graphs/median_c_540000_r_5.png")
