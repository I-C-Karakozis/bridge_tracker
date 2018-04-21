import numpy as np 
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from tools.general_purpose import *

labels = collect_all_files("data/labels")
print(len(labels))

counts = np.zeros(11)
counts_no_dummy = np.zeros(11)
index = list(range(1,11))

for file in labels:
    # load frame and ground truth 
    with open(file) as f:
        boxes = f.readlines() 
    counts[int(boxes[0]) - 1] = counts[int(boxes[0]) - 1] + 1

    instances = int(boxes[0])
    for box in boxes:
        xy = box.split()
        label = xy[-1]
        
        if label == "Dummy":
            instances = instances - 1

    counts_no_dummy[instances-1] = counts_no_dummy[instances-1] + 1

plt.plot(index, counts[:10] / np.sum(counts) * 100, '-bo')
plt.plot(index, counts_no_dummy[:10] / np.sum(counts) * 100, '-ro')
plt.title("Number of Object Instances per Frame")
plt.xlabel("Number of instances")
plt.ylabel("Percentage of frames")
plt.legend(["Cards + Dummy", "Cards"])
plt.show()
