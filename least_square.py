import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from data_process import *
from implementations import *
from helpers import *


y, tX, ids = load_clean_data()

### grid search for hyperparameters gamma and initial_w

max_iters = 50
gammas = np.logspace(-10, -4)
initial_ws = np.array([np.random.rand(tX.shape[1]) * n for n in np.logspace(-3, 3)])
losses = np.full((len(gammas), len(initial_ws)), -1)

for i in range(len(gammas)):
	for j in range(len(initial_ws)):
		gamma = gammas[i]
		w_initial = initial_ws[j]
		weights, loss = least_squares_SGD(y, tX, w_initial, max_iters, gamma)
		if loss != np.inf:
			losses[i, j] = loss

### heatmap


fig, ax = plt.subplots()
im = ax.imshow(losses)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(cbarlabel='', rotation=-90, va="bottom")


# We want to show all ticks...
ax.set_xticks(np.arange(len(gammas)))
ax.set_yticks(np.arange(len(initial_ws)))
# ... and label them with the respective list entries
ax.set_xticklabels(gammas)
ax.set_yticklabels(initial_ws)

# Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
# 		rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
# for i in range(len(initial_ws)):
# 	for j in range(len(gammas)):
# 		text = ax.text(j, i, losses[i, j],
# 						ha="center", va="center", color="w")


# Turn spines off and create white grid.
for edge, spine in ax.spines.items():
	spine.set_visible(False)

ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

# ax.set_title("losses of local gammas (in tons/year)")
fig.tight_layout()
plt.show()


