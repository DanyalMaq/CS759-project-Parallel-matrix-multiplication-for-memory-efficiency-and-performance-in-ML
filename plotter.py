import os
import matplotlib.pyplot as plt

managed = [] # [single, max of 2]
manual = [] # [single, max of 2]
asy = [] # [max of 2]

with open("./values.txt") as f:
	lines = f.readlines()
	for i in range(10):
		m1 = float(lines[i*13 + 2].split(" = ")[1])
		m2 = max(float(lines[i*13 + 3].split(" ")[-2]), float(lines[i*13 + 3].split(" ")[-2]))
		managed.append([m1, m2])

		m2 = max(float(lines[i*13+6].split(" ")[-2]), float(lines[i*13+7].split(" ")[-2]))
		m1 = float(lines[i*13+8].split(" ")[-1])
		manual.append([m1, m2])

		a = max(float(lines[i*13+10].split(" ")[-2]), float(lines[i*13+11].split(" ")[-2]))
		asy.append(a)

x = [2**i for i in range(5, 15)]
managed_single = [n[0] for n in managed]
managed_multi = [n[1] for n in managed]
manual_single = [n[0] for n in manual]
manual_multi = [n[1] for n in manual]

# PLots Times
# plt.plot(x, managed_single, marker='o', linestyle='-', color='blue', label='Managed (Single GPU)')
# plt.plot(x, managed_multi, marker='o', linestyle='--', color='blue', label='Managed (max time on a GPU)')
# plt.plot(x, manual_single, marker='s', linestyle='-', color='green', label='Manual (Single GPU)')
# plt.plot(x, manual_multi, marker='s', linestyle='--', color='green', label='Manual (max time on a GPU)')
# plt.plot(x, asy, marker='^', linestyle='--', color='red', label='Async (max time on a GPU)')

# plt.xlabel('Powers of 2')
# plt.ylabel('Logarithmic time (ms)')
# plt.title('Matmul performance with\n Managed, Manual, and Asynchronously allocated Arrays')
# plt.xscale('log', base=2) 
# plt.yscale('log', base=10)
# plt.legend()

# plt.savefig("results.png")

# Plots increasing GPUs
plt.plot([2, 3, 4], [112.67, 119, 116], marker='s', linestyle='-', color='blue', label='Time taken on a single GPU')
plt.plot([2, 3, 4], [62.3, 41.3, 31.66], marker='o', linestyle='--', color='green', label='Max time on a GPU among multiple GPUs')

plt.xlabel('Number of GPUs')
plt.ylabel('Time (ms)')
plt.title('Matmul performance with multiple GPUs')
plt.legend()

plt.savefig("all_gpu.png")