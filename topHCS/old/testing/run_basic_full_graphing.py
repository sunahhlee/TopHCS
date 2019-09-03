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

# graphing initials

n = 3
colors = plt.cm.plasma(np.linspace(0,0.95,n))

fig, axs = plt.subplots(2, 4, figsize=(40,16))
fontsize=17

device = 'cuda' 

accumVecPaths = -1
kVals = [50000, 100000]
hVals = [0.5, 1.0]
rVals = [5, 15]
c = 180000

csvec_accuracy = [[[0.1543, 0.09229], [0.3211, 0.19329]], [[0.00656, 0.00856], [0.0379, 0.03852]]]
csvec_L2 = [[[1.71890879e-05, 2.16076169e-05], [7.52577307e-06, 1.03917146e-05]], [[0.01835084, 0.02452415], [0.01133576, 0.01507866]]]

half_accuracy = [[[0.21412, 0.12725], [0.37652, 0.24214]], [[0.00744, 0.00982], [0.0417, 0.044]]]
half_L2 = [[[1.11603767e-05, 1.46940065e-05], [6.48265086e-06, 8.79577055e-06]], [[0.01785514,0.02350082], [0.0110609, 0.014491]]]

full_L2 = [[[1.07820888e-05, 1.40615657e-05], [6.25502980e-06, 8.43287489e-06]], [[0.0175926, 0.02290335], [0.01090221, 0.01415622]]]
full_accuracy = [[[0.22136, 0.13394], [0.39184, 0.25581]], [[0.00752, 0.0105], [0.0434, 0.04697]]]

for p, paths in enumerate(["initial", "accum"]):
	for r_i, rows in enumerate(rVals):
		r = rows 
		axs[p, 2*r_i].tick_params(labelsize=fontsize-2)
		axs[p, 2*r_i].set_xlabel("k", fontsize=fontsize-2)
		axs[p, 2*r_i].set_ylabel("index accuracy", fontsize=fontsize-2)
		axs[p, 2*r_i].set_title("index accuracy vs. k, sketch=("+str(r)+", "+str(c)+")", fontsize=fontsize)	
		axs[p, 2*r_i].plot(kVals, csvec_accuracy[p][r_i], color=colors[0], label="CSVec") 
		axs[p, 2*r_i].plot(kVals, half_accuracy[p][r_i], color=colors[1], label="h = 0.5*k")
		axs[p, 2*r_i].plot(kVals, full_accuracy[p][r_i], color=colors[2], label="h = k")
		axs[p, 2*r_i].legend(fontsize=fontsize-3)

		axs[p, 2*r_i + 1].tick_params(labelsize=fontsize-2)
		axs[p, 2*r_i + 1].set_xlabel("k", fontsize=fontsize-2)
		axs[p, 2*r_i + 1].set_ylabel("L2 norm = sqrt(sum((expected - recovered)^2))", fontsize=fontsize-2)
		axs[p, 2*r_i + 1].set_title("L2 norm vs. k, sketch=("+str(r)+", "+str(c)+")", fontsize=fontsize)
		axs[p, 2*r_i + 1].plot(kVals, csvec_L2[p][r_i], color=colors[0], label="CSVec")
		axs[p, 2*r_i + 1].plot(kVals, half_L2[p][r_i], color=colors[1], label="h = 0.5*k")
		axs[p, 2*r_i + 1].plot(kVals, full_L2[p][r_i], color=colors[2], label="h = k")
		axs[p, 2*r_i + 1].legend(fontsize=fontsize-3)
plt.savefig("graphs/basic_full_numBlocks_40.png")
