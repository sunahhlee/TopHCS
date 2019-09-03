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

n = 4
colors = plt.cm.plasma(np.linspace(0,0.95,n))

fig, axs = plt.subplots(2, 4, figsize=(38,16))
fontsize=17

device = 'cuda' 
initialVecPaths = ["../../datafiles/initialVs0.torch", 
                   "../../datafiles/initialVs1.torch", 
                   "../../datafiles/initialVs2.torch", 
                   "../../datafiles/initialVs3.torch"]
accumVecPaths = ["../../datafiles/accumulatedVs0.torch", 
                   "../../datafiles/accumulatedVs1.torch", 
                   "../../datafiles/accumulatedVs2.torch", 
                   "../../datafiles/accumulatedVs3.torch"]
for p, paths in enumerate([accumVecPaths]):
	vecs = []
	for path in paths:
		vecs.append(torch.load(path, map_location=device))
	assert(len(vecs) == len(initialVecPaths))
	print("Loaded paths : ", paths)

	summed = vecs[0].clone()
	for v in vecs[1:]:
		summed += v

	assert(summed.size() == vecs[0].size())

	expected = torch.zeros(len(summed), device=device)
	expectedIndices = torch.sort(summed**2)[1]

	kVals = [50000, 100000, 150000]
	kVals = [100000]
	hVals = [0, .50, 1.00]
	rVals = [5, 15]
	rVals = [15]
	
	for r_i, rows in enumerate(rVals):
		csvec_accuracy = np.zeros(len(kVals))
		topHCS_accuracy = np.zeros((len(hVals), len(kVals)))
		csvec_L2 = np.zeros(len(kVals))
		topHCS_L2 = np.zeros((len(hVals), len(kVals)))
		for k_i, k in enumerate(kVals): 
			d, c, r, numBlocks = len(summed), 180000, rows, 40
			print("d = ", d, '\n', "k = ", k)

			expected[expectedIndices[-k:].to(device)] = summed[expectedIndices[-k:].to(device)]

			w_0 = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w_0 += vecs[0]

			w_1 = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w_1 += vecs[1]

			w_2 = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w_2 += vecs[2]

			w_3 = CSVec(d=d, c=c, r=r, numBlocks=numBlocks, device=device)
			w_3 += vecs[3]

			workers_summed = w_0 + w_1 + w_2 + w_3
			result = workers_summed.unSketch(k)

			#csvec_accuracy = (expected == result).nonzero().numel() / k
			csvec_accuracy[k_i] = (expected[expectedIndices] * result[expectedIndices]).nonzero().numel() / k
			#print('expected :', expected)
			#print('result :', result)
			squared_diff = (expected - result)**2
			csvec_L2[k_i] = (torch.sum(squared_diff))**0.5


			for h_i, h in enumerate(hVals):
				d, c, r, numBlocks = len(summed), 180000, rows, 40
				h_curr = int(h*k)
				#print("d = ", d, '\n', "k = ", k, "h : ", h_curr)
				w_0 = TopHCS(d=d, c=c, r=r, h=h_curr, numBlocks=numBlocks, device=device)
				w_0.store(vecs[0])
				print("H worker added")
                
				w_1 = TopHCS(d=d, c=c, r=r, h=h_curr, numBlocks=numBlocks, device=device)
				w_1.store(vecs[1])
				print("H worker added")

				w_2 = TopHCS(d=d, c=c, r=r, h=h_curr, numBlocks=numBlocks, device=device)
				w_2.store(vecs[2])

				print("H worker added")
				w_3 = TopHCS(d=d, c=c, r=r, h=h_curr, numBlocks=numBlocks, device=device)
				w_3.store(vecs[3])

				print("H worker added")
				workers = [w_0, w_1, w_2, w_3]
				result = TopHCS.topKSum(workers, k)
				 
				#topHCS_accuracy = (expected == result).nonzero().numel() / k
				topHCS_accuracy[h_i, k_i]  = (expected[expectedIndices] * result[expectedIndices]).nonzero().numel() / k
				#print('expected :', expected)
				#print('result :', result)
				squared_diff = (expected - result)**2
				topHCS_L2[h_i, k_i] = (torch.sum(squared_diff))**0.5

		axs[p, 2*r_i].tick_params(labelsize=fontsize-2)
		axs[p, 2*r_i].set_xlabel("k", fontsize=fontsize-2)
		axs[p, 2*r_i].set_ylabel("index accuracy", fontsize=fontsize-2)
		axs[p, 2*r_i].set_title("index accuracy vs. k, sketch=("+str(r)+", "+str(c)+")", fontsize=fontsize)	
		axs[p, 2*r_i].plot(kVals, csvec_accuracy, color=colors[0], label="CSVec") 
		for h_i, h in enumerate(hVals):
			axs[p, 2*r_i].plot(kVals, topHCS_accuracy[h_i], color=colors[h_i+1], label="h = k*"+str(h))
		axs[p, 2*r_i].legend(fontsize=fontsize-3)

		axs[p, 2*r_i + 1].tick_params(labelsize=fontsize-2)
		axs[p, 2*r_i + 1].set_xlabel("k", fontsize=fontsize-2)
		axs[p, 2*r_i + 1].set_ylabel("L2 norm = sqrt(sum((expected - recovered)^2))", fontsize=fontsize-2)
		axs[p, 2*r_i + 1].set_title("L2 norm vs. k, sketch=("+str(r)+", "+str(c)+")", fontsize=fontsize)
		axs[p, 2*r_i + 1].plot(kVals, csvec_L2, color=colors[0], label="CSVec")
		for h_i, h in enumerate(hVals):
			axs[p, 2*r_i + 1].plot(kVals, topHCS_L2[h_i], color=colors[h_i+1], label="h = k*"+str(h))
		axs[p, 2*r_i + 1].legend(fontsize=fontsize-3)
		print("h values : ", hVals)
		print("k values : ", kVals)
		print("row : ", r)
		print("CSVec accuracy : ", csvec_accuracy)
		print("Top HCS accuracy : ", '\n', topHCS_accuracy)
		print("CSVec L2 : ", csvec_L2)
		print("Top HCS L2 : ", topHCS_L2)
plt.savefig("graphs/basic_full_numBlocks_40.png")
