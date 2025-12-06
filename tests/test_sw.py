import torch
import pytest
import numpy as np
from models.ctm import ContinuousThoughtMachine, generate_hub_spoke_topology
from utils.housekeeping import set_seed
import networkx as nx

# --- FIXTURES ---

@pytest.fixture
def sw_config():
    """Standard configuration for testing."""
    return {
        "d_model": 1024,
        "n_synch": 128,
    }

@pytest.fixture
def model_factory():
    """Creates a small, fast CTM for integration tests."""
    def create_model(neuron_select_type='small-world'):
        return ContinuousThoughtMachine(
            d_model=128,
            n_synch_out=32,
            n_synch_action=32,
            iterations=5,
            d_input=32,
            heads=2,
            synapse_depth=1,
            memory_length=5,
            deep_nlms=False,
            memory_hidden_dims=16,
            do_layernorm_nlm=False,
            backbone_type='none',
            positional_embedding_type='none',
            out_dims=10,
            neuron_select_type=neuron_select_type
        )
    return create_model

# --- TOPOLOGY PROPERTIES ---

def test_topology_budget_exactness(sw_config):
    """
    Property: The generator must return EXACTLY n_synch edges.
    Reasoning: CTM relies on fixed tensor shapes for compilation and batching.
    """
    left, right, decay, types = generate_hub_spoke_topology(**sw_config)
    
    assert len(left) == sw_config['n_synch'], f"Budget mismatch: Got {len(left)}, expected {sw_config['n_synch']}"
    assert len(right) == sw_config['n_synch']
    assert len(decay) == sw_config['n_synch']
    assert len(types) == sw_config['n_synch']

def test_topology_valid_indices(sw_config):
    """
    Property: All indices must be within [0, d_model).
    """
    left, right, _, _ = generate_hub_spoke_topology(**sw_config)
    assert left.min() >= 0 and left.max() < sw_config['d_model']
    assert right.min() >= 0 and right.max() < sw_config['d_model']

def test_hub_structure_and_types(sw_config):
    """
    Property:
    1. Hubs must exist (min 2 for a ring).
    2. Type 0 edges must contain BOTH Self-Loops AND Lateral Connections.
    3. Self-Loops must be exactly N_hubs.
    4. Lateral edges must be AT LEAST N_hubs (Ring) + Shortcuts.
    """
    left, right, _, types = generate_hub_spoke_topology(**sw_config)
    
    # Filter for Type 0 (Core = Memory + Lateral)
    core_mask = (types == 0)
    core_left = left[core_mask]
    core_right = right[core_mask]
    
    # Separate Self-Loops (Memory) from Lateral (Ring + Shortcuts)
    self_loops_mask = (core_left == core_right)
    cross_edges_mask = (core_left != core_right)
    
    num_self_loops = self_loops_mask.sum().item()
    num_lateral_edges = cross_edges_mask.sum().item()

    # Check 1: Hub Count
    expected_hubs = max(2, int(sw_config['n_synch'] * 0.1))
    
    # Verify exactly one self-loop per hub (Memory Register)
    assert num_self_loops == expected_hubs, f"Expected {expected_hubs} self-loops, found {num_self_loops}"
    
    # Verify lateral connections (Ring + Shortcuts)
    # Must have AT LEAST the ring edges
    assert num_lateral_edges >= expected_hubs, f"Expected at least {expected_hubs} lateral edges, found {num_lateral_edges}"

def test_feeder_connectivity_structure(sw_config):
    """
    Property:
    1. Feeders (Type 1) must connect Periphery -> Hub.
    2. Feeders MUST NOT connect Periphery -> Periphery.
    """
    left, right, _, types = generate_hub_spoke_topology(**sw_config)
    
    # Hubs: They are the nodes that possess a Type 0 Self-Loop
    core_mask = (types == 0)
    # Find indices where left == right in the core set
    hubs = set(left[core_mask][left[core_mask] == right[core_mask]].tolist())
    
    # Analyze Feeders (Type 1)
    feeder_mask = (types == 1)
    feeder_sources = left[feeder_mask].tolist()
    feeder_targets = right[feeder_mask].tolist()
    
    # Verify every feeder target is a Hub
    for t in feeder_targets:
        assert t in hubs, f"Feeder target {t} is not a Hub! Feeders must target Memory."
        
    # Verify no feeder source is a Hub (Strict Periphery -> Hub)
    for s in feeder_sources:
        assert s not in hubs, f"Feeder source {s} is a Hub! Hubs should not feed other Hubs via Type 1 edges."

def test_hub_ring_topology(sw_config):
    """
    Property: Hubs must form a Strongly Connected component via Type 0 lateral edges.
    Reasoning: Ensures information can propagate from any Hub to any other Hub (Global Workspace),
               even if shortcuts create branches/cycles.
    """
    left, right, _, types = generate_hub_spoke_topology(**sw_config)
    
    # Get Type 0 Lateral Edges (Ring + Shortcuts)
    core_mask = (types == 0)
    cross_mask = (left[core_mask] != right[core_mask])
    
    lateral_sources = left[core_mask][cross_mask].tolist()
    lateral_targets = right[core_mask][cross_mask].tolist()
    
    # Construct Graph
    G = nx.DiGraph()
    edges = list(zip(lateral_sources, lateral_targets))
    G.add_edges_from(edges)
    
    # Verify 1: All Hubs are in the graph
    core_left = left[core_mask]
    core_right = right[core_mask]
    hubs = set(core_left[core_left == core_right].tolist())
    assert set(G.nodes()) == hubs, "Lateral graph has disconnected/missing hubs!"

    # Verify 2: Strong Connectivity
    # In a Ring (or Small-World), every node is reachable from every other node.
    assert nx.is_strongly_connected(G), "Topology is fragmented! Information cannot circulate globally."

    # Verify 3: Minimum Connectivity (Ring Property)
    # Every Hub must have at least one outgoing and one incoming connection
    for node in G.nodes():
        assert G.out_degree(node) >= 1, f"Hub {node} is a sink (dead end)!"
        assert G.in_degree(node) >= 1, f"Hub {node} is a source (unreachable)!"

# --- PHYSICS & DYNAMICS ---

def test_decay_initialization_distributions(sw_config):
    """
    Property:
    1. Core (Type 0) -> Slow Decay (~0.1)
    2. Feeder (Type 1) -> Fast Decay (~0.8)
    """
    _, _, decay, types = generate_hub_spoke_topology(**sw_config)
    
    core_decay = decay[types == 0]
    feeder_decay = decay[types == 1]
    
    # Check Core Statistics
    # Mean ~ 0.1, allow some statistical wiggle room
    assert 0.0 < core_decay.mean() < 0.2, f"Core decay too fast/slow: {core_decay.mean()}"
    
    # Check Feeder Statistics
    # Mean ~ 0.8
    assert 0.6 < feeder_decay.mean() < 1.0, f"Feeder decay too fast/slow: {feeder_decay.mean()}"
    
    # Check Separation
    # Ensure the bulk of the distributions do not overlap
    core_99 = torch.quantile(core_decay, 0.99)
    feeder_01 = torch.quantile(feeder_decay, 0.01)
    assert core_99 < feeder_01, "Decay distributions overlap! Functional segregation compromised."

# --- 4. MODEL INTEGRATION ---

def test_forward_pass_integrity(model_factory):
    """
    Property: The model must run a full forward pass without shape errors.
    """
    model = model_factory(neuron_select_type='small-world')
    x = torch.randn(2, 32, 39, 39)
    
    model.train()
    try:
        preds, certs, synch = model(x)
    except Exception as e:
        pytest.fail(f"Forward pass crashed: {e}")
    else:
        loss = preds.sum() # We optimize a dummy loss
        loss.backward()
        
    assert preds.shape == (2, 10, 5) # (B, Out, Iters)
    assert not torch.isnan(preds).any()

    # 'start_trace' is the parameter that initializes the history buffer.
    # If the gradient reaches here, it means it successfully flowed back 
    # through all 'torch.stack' and 'list.append' operations in the loop.
    assert model.start_trace.grad is not None, "Gradient flow broken: 'start_trace' received no gradient."

def test_determinism(sw_config):
    """
    Property: Setting seed must produce identical topologies.
    """
    set_seed(42)
    l1, r1, d1, t1 = generate_hub_spoke_topology(**sw_config)
    
    set_seed(42)
    l2, r2, d2, t2 = generate_hub_spoke_topology(**sw_config)
    
    assert torch.equal(l1, l2)
    assert torch.equal(r1, r2)
    assert torch.equal(d1, d2)
    assert torch.equal(t1, t2)