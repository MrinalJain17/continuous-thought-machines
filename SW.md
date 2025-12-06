# Optimizing Neural Dynamics: A Small-World Topology for Continuous Thought Machines

Biological brains balance local specialization with global integration using Small-World networks. The current CTM baseline for navigation tasks relies on "Dense Pairing," creating a computationally convenient but topologically inefficient clique. This experiment replaces that clique with a **Hub-and-Spoke Small-World topology**, aiming to improve sequential reasoning and generalization without increasing the parameter budget.

### The Bottleneck: Dense Pairing
In the baseline Maze task, the synchronization mechanism selects a dense clique of just **32 neurons** out of a latent space of 2048. While this captures high-order correlations within that small subset, it effectively ignores the remaining 2016 neurons. This forces the model to compress the entire "world state" (the maze, the path, the plan) into a tiny bottleneck (1.5% of the latent space), potentially limiting its ability to form a robust cognitive map.

### The Solution: Hub-and-Spoke Topology
We propose a structural intervention: maintaining the exact same parameter budget (528 pairs) but distributing it through a **Small-World (Hub-and-Spoke)** topology.

Instead of 32 neurons talking only to each other, we select roughly **16 "Hub" neurons** that act as global broadcasters. By trading "node count" for "receptive field," each Hub connects to ~31 distinct neighbors across the latent space, expanding the synchronization layer's visibility from <2% to near 100% of the manifold.

### Structural Priors: The "Three-Tier" Logic
We move beyond random initialization by baking biological priors directly into the graph structure and decay parameters ($r_{ij}$).

1.  **Identity (Self-Pairing):** Every Hub connects to itself.
    * **Logic:** Covariance alone tracks alignment, not intensity. Self-pairing provides the "energy" (variance) needed for proper normalization.
    * **Decay:** Initialized to **0.0** (Infinite Memory) to anchor the neuron's identity throughout the thought process.

2.  **Memory (Lattice Connections):** Hubs connect to their local spatial neighbors in the latent ring.
    * **Logic:** Local features (e.g., adjacent maze walls) are spatially consistent and require stability.
    * **Decay:** Initialized to **~0.1** (Long Working Memory).

3.  **Transient Events (Rewired Shortcuts):** A fraction ($p=0.2$) of connections are rewired to random distant neurons.
    * **Logic:** Long-range connections represent semantic jumps or global insights. These are "events" rather than states.
    * **Decay:** Initialized to **~1.0** (Transient/Coincidence Detection).



### Implementation & Constraints
The new topology is generated procedurally during initialization using a vectorized approach that is fully compatible with the existing CTM architecture. Crucially, the generator is **budget-aware**:

* **Baseline (Control):** 32 Neurons $\times$ Dense Connectivity = **528 Pairs**.
* **Experiment (Treatment):** 16 Hubs $\times$ 32 Connections (Self + Neighbors) = **528 Pairs**.

By aligning the synchronization vector size, we ensure that the readout layer ($W_{out}$) has the exact same number of learnable parameters. Any performance gain is purely attributable to superior topology, not increased capacity.

### The Hypothesis
We expect the Small-World CTM to outperform the Dense baseline in two key areas:
1.  **Convergence Speed:** Gradients can flow from any part of the 2048-neuron latent space to the readout layer via the Hubs, eliminating the "blind spot" problem.
2.  **Generalization:** The structural distinction between "Local Memory" and "Global Shortcuts" maps perfectly to sequential planning tasks, where the agent must maintain a stable history (path) while scanning for distant objectives (exit).
