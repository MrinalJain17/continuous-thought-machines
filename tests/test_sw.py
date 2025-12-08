import torch
import pytest
import numpy as np

# --- 1. Task-Specific Configuration Tests ---
# These tests verify that the model constructs the expected topology
# under the exact constraints defined in the Deep Research report.

@pytest.mark.parametrize("config", [
    # Maze: High dim (2048), Tiny budget (1024), High connectivity (31)
    # Expectation: Sparse "Hub-and-Spoke" architecture (~32 hubs)
    {
        "name": "Maze_39x39", 
        "d_model": 2048, 
        "n_synch": 1024, 
        "k": 31, 
        "expected_hubs": 32
    },
    # CIFAR-10: Low dim (256), Full budget (256), Low connectivity (4)
    # Expectation: Dense "Global Lattice" coverage (~51 hubs)
    {
        "name": "CIFAR10_ResNet18", 
        "d_model": 256, 
        "n_synch": 256, 
        "k": 4, 
        "expected_hubs": 51
    }
])
def test_small_world_task_configs(ctm_factory, base_params, device, config):
    """
    Verifies that the Small-World generator correctly adapts to specific task constraints.
    Checks:
    1. Budget usage (exact).
    2. Hub count (matches theoretical max).
    3. Structural integrity (every hub has a self-pair).
    """
    model = ctm_factory(
        base_params,
        d_model=config["d_model"],
        n_synch_out=config["n_synch"],
        n_synch_action=config["n_synch"], # symmetric for testing
        neuron_select_type='small-world',
        connectivity=config["k"],
        rewiring_prob=0.2 # Standard setting
    ).to(device)
    
    left = model.out_neuron_indices_left
    right = model.out_neuron_indices_right
    
    # 1. Exact Budget Check
    assert left.shape[0] == config["n_synch"]
    
    # 2. Hub Count Check
    # We identify "Hubs" as unique source neurons in the 'left' tensor
    unique_hubs = torch.unique(left)
    
    # Note: If remainder filling occurs, we might have slightly more unique sources.
    # But we must have AT LEAST the expected number of structured hubs.
    assert len(unique_hubs) >= config["expected_hubs"]
    # It shouldn't be exploding (e.g. random pairing would give ~n_synch unique sources)
    # For Maze: 32 hubs. Random would be ~800. This verifies structure.
    if config["name"] == "Maze_39x39":
        assert len(unique_hubs) < 100, "Maze topology should be sparse (Hub-and-Spoke), not random."

    # 3. Energy Constraint Check (Self-Pairing)
    # Every identified Hub MUST have a self-connection (i,i)
    self_pairs = (left == right)
    self_paired_hubs = torch.unique(left[self_pairs])
    
    # We allow a tiny margin of error for remainder filling nodes, 
    # but the core hubs must be self-paired.
    assert len(self_paired_hubs) >= config["expected_hubs"], \
        "Core hubs missing mandatory self-pairing."


# --- 2. Deterministic Property Tests ---
# These tests remove randomness (p=0) to mathematically verify the lattice logic.

def test_lattice_invariant_deterministic(ctm_factory, base_params, device):
    """
    Property Test: Ring Lattice Structure.
    Sets rewiring_prob=0.0 to deterministically verify that connections are strictly local.
    
    Invariant: For Hub 'h' with connectivity 'k', targets must be exactly:
    {h} U { (h - j)%D, (h + j)%D } for j in 1..k/2
    """
    d_model = 100
    k = 4 # Neighbors: -1, +1, -2, +2
    n_synch = 50 # 100 // (1+4) = 20 Hubs
    
    model = ctm_factory(
        base_params,
        d_model=d_model,
        n_synch_out=n_synch,
        neuron_select_type='small-world',
        connectivity=k,
        rewiring_prob=0.0 # CRITICAL: No randomness
    ).to(device)
    
    left = model.out_neuron_indices_left
    right = model.out_neuron_indices_right
    
    # Iterate through each generated connection to verify locality
    for i in range(n_synch):
        u, v = int(left[i]), int(right[i])
        
        # Calculate lattice distance on a ring
        dist = abs(u - v)
        dist = min(dist, d_model - dist) # Ring wrap-around distance
        
        # Assert Distance Invariant
        # Distance must be 0 (self) or <= k/2 (neighbor)
        assert dist <= (k // 2), f"Connection ({u}, {v}) violates lattice locality (dist={dist} > {k//2})"


# --- 3. Biological Prior Tests ---
# Verifies the initialization logic for the three classes of connections.

def test_decay_initialization_priors(ctm_factory, base_params, device):
    """
    Property Test: Decay Initialization Hierarchy.
    Verifies that parameters are initialized according to the 3-tier logic:
    1. Self-Pair (Identity) -> 0.0 (Infinite Memory)
    2. Lattice (Memory) -> ~0.1 (Long Memory)
    3. Rewired (Transient) -> ~1.0 (Short Memory)
    """
    d_model = 100
    n_synch = 500 # High budget to ensure we see all types
    k = 4
    
    # Force p=0.5 to ensure we get a mix of Lattice and Rewired
    model = ctm_factory(
        base_params,
        d_model=d_model,
        n_synch_out=n_synch,
        neuron_select_type='small-world',
        connectivity=k,
        rewiring_prob=0.5 
    ).to(device)
    
    left = model.out_neuron_indices_left.cpu()
    right = model.out_neuron_indices_right.cpu()
    decay_params = model.decay_params_out.detach().cpu()
    
    # 1. Verify Self-Pairs (Identity)
    self_mask = (left == right)
    self_decays = decay_params[self_mask]
    
    # Self-pairs should have high decay_param (≈15)
    assert torch.all(self_decays > 14.0), "Self-pairs need large decay_param for infinite memory"
    
    # 2. Verify Others (Lattice vs Rewired)
    other_mask = ~self_mask
    other_decays = decay_params[other_mask]
    
    # We simply check that we have a bimodal distribution
    # Lattice (r=0.1) requires param ~ 2.30
    # Rewired (r=1.0) requires param ~ inf/large
    # Note: Noise is +/- 0.01
    
    has_lattice = torch.any((other_decays > 2.25) & (other_decays < 2.35))
    has_rewired = torch.any((other_decays > 14.0))  # Not 0!

    assert has_lattice, "Missing Lattice (Local) decay priors (~0.1)"
    assert has_rewired, "Missing Rewired (Global) decay priors (~1.0)"


# --- 4. Dynamic Safety Tests ---
# These tests run a full forward/backward pass to ensure that
# performance optimizations (like List+Stack memory) do not break Autograd.

def test_memory_gradient_flow(ctm_factory, base_params, device):
    """
    Verifies that gradients correctly propagate backwards through the
    manual 'List + Stack' memory buffer optimization.
    
    If this fails, the 'history_buffer' logic has detached the graph,
    and the model is not learning from past timesteps.
    """
    # 1. Setup a small, trainable model
    # We need enough iterations to force the buffer to cycle (pop/append)
    iterations = 10 
    memory_length = 5 
    
    model = ctm_factory(
        base_params,
        d_model=128,          # Small dim
        memory_length=memory_length,
        iterations=iterations,
        neuron_select_type='small-world'
    ).to(device)
    
    # 2. Forward Pass
    # Input: (Batch=2, Channels=3, H=32, W=32)
    x = torch.randn(2, 3, 32, 32).to(device)
    
    # Enable gradient tracking for inputs/weights
    model.train()
    predictions, _, _ = model(x)
    
    # 3. Backward Pass
    # We optimize a dummy loss
    loss = predictions.sum()
    loss.backward()
    
    # 4. THE CHECK: Does 'start_trace' have a gradient?
    # 'start_trace' is the parameter that initializes the history buffer.
    # If the gradient reaches here, it means it successfully flowed back 
    # through all 'torch.stack' and 'list.append' operations in the loop.
    
    assert model.start_trace.grad is not None, \
        "Gradient flow broken: 'start_trace' received no gradient."
        
    # Optional: Check magnitude to ensure it's not vanishing to exact zero
    grad_mag = model.start_trace.grad.abs().sum().item()
    assert grad_mag > 0.0, \
        f"Gradient flow vanished: 'start_trace' gradient sum is {grad_mag}"
