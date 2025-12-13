
import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import imageio
import seaborn as sns
import networkx as nx
from tqdm.auto import tqdm

def find_center_of_mass(array_2d):
    """
    Alternative implementation using np.average and meshgrid.
    This version is generally faster and more concise.

    Args:
        array_2d: A 2D numpy array of values between 0 and 1.

    Returns:
        A tuple (x, y) representing the coordinates of the center of mass.
    """
    total_mass = np.sum(array_2d)
    if total_mass == 0:
      return (np.nan, np.nan)

    y_coords, x_coords = np.mgrid[:array_2d.shape[0], :array_2d.shape[1]]
    x_center = np.average(x_coords, weights=array_2d)
    y_center = np.average(y_coords, weights=array_2d)
    return (round(y_center, 4), round(x_center, 4))

def draw_path(x, route, valid_only=False, gt=False, cmap=None):
    """
    Draws a path on a maze image based on a given route.

    Args:
        maze: A numpy array representing the maze image.
        route: A list of integers representing the route, where 0 is up, 1 is down, 2 is left, and 3 is right.
        valid_only: A boolean indicating whether to only draw valid steps (i.e., steps that don't go into walls).

    Returns:
        A numpy array representing the maze image with the path drawn in blue.
    """
    x = np.copy(x)
    start = np.argwhere((x == [1, 0, 0]).all(axis=2))
    end = np.argwhere((x == [0, 1, 0]).all(axis=2))
    if cmap is None:
        cmap = plt.get_cmap('winter') if not valid_only else  plt.get_cmap('summer')

    # Initialize the current position
    current_pos = start[0]

    # Draw the path
    colors = cmap(np.linspace(0, 1, len(route)))
    si = 0
    for step in route:
        new_pos = current_pos
        if step == 0:  # Up
            new_pos = (current_pos[0] - 1, current_pos[1])
        elif step == 1:  # Down
            new_pos = (current_pos[0] + 1, current_pos[1])
        elif step == 2:  # Left
            new_pos = (current_pos[0], current_pos[1] - 1)
        elif step == 3:  # Right
            new_pos = (current_pos[0], current_pos[1] + 1)
        elif step == 4:  # Do nothing
            pass
        else:
            raise ValueError("Invalid step: {}".format(step))

        # Check if the new position is valid
        if valid_only:
            try:
                if np.all(x[new_pos] == [0,0,0]):  # Check if it's a wall
                    continue  # Skip this step if it's invalid
            except IndexError:
                continue  # Skip this step if it's out of bounds

        # Draw the step
        if new_pos[0] >= 0 and new_pos[0] < x.shape[0] and new_pos[1] >= 0 and new_pos[1] < x.shape[1]:
            if not ((x[new_pos] == [1,0,0]).all() or (x[new_pos] == [0,1,0]).all()):
                colour = colors[si][:3]
                si += 1
                x[new_pos] = x[new_pos]*0.5 + colour*0.5

        # Update the current position
        current_pos = new_pos
        # cv2.imwrite('maze2.png', x[:,:,::-1]*255)

    return x

def make_maze_gif(inputs, predictions, targets, attention_tracking, save_location, verbose=True):
    """
    Expect inputs, predictions, targets as numpy arrays
    """
    route_steps = []
    route_colours = []
    solution_maze = draw_path(np.moveaxis(inputs, 0, -1), targets)
    
    n_heads = attention_tracking.shape[1]
    mosaic = [['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['head_0', 'head_1', 'head_2', 'head_3', 'head_4', 'head_5', 'head_6', 'head_7'],
              ['head_8', 'head_9', 'head_10', 'head_11', 'head_12', 'head_13', 'head_14', 'head_15'],
              ]
    if n_heads == 8:
        mosaic = [['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['head_0', 'head_1', 'head_2', 'head_3', 'head_4', 'head_5', 'head_6', 'head_7'],
              ]
    elif n_heads == 4:
        mosaic = [['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['overlay', 'overlay', 'overlay', 'overlay', 'route', 'route', 'route', 'route'],
              ['head_0', 'head_0', 'head_1', 'head_1', 'head_2', 'head_2', 'head_3', 'head_3'],
              ['head_0', 'head_0', 'head_1', 'head_1', 'head_2', 'head_2', 'head_3', 'head_3'],
              ]

    img_aspect = 1
    figscale = 1
    aspect_ratio = (len(mosaic[0]) * figscale, len(mosaic) * figscale * img_aspect) # W, H
    
    route_steps = [np.unravel_index(np.argmax((inputs == np.reshape(np.array([1, 0, 0]), (3, 1, 1))).all(0)), inputs.shape[1:])]  # Starting point
    frames = []
    cmap = plt.get_cmap('gist_rainbow')
    cmap_viridis = plt.get_cmap('viridis')
    step_linspace = np.linspace(0, 1, predictions.shape[-1])  # For sampling colours
    with tqdm(total=predictions.shape[-1], initial=0, leave=False, position=1, dynamic_ncols=True) as pbar:
        if verbose: pbar.set_description('Processing frames for maze plotting')
        for stepi in np.arange(0, predictions.shape[-1], 1):
            fig, axes = plt.subplot_mosaic(mosaic, figsize=aspect_ratio)
            for ax in axes.values():
                ax.axis('off')
            guess_maze = draw_path(np.moveaxis(inputs, 0, -1), predictions.argmax(1)[:,stepi], cmap=cmap)
            attention_now = attention_tracking[stepi]
            for hi in range(min((attention_tracking.shape[1], 16))):
                ax = axes[f'head_{hi}']
                attn = attention_tracking[stepi, hi]
                attn = (attn - attn.min())/(np.ptp(attn))
                ax.imshow(attn, cmap=cmap_viridis)
            # Upsample attention just for visualisation
            aggregated_attention = torch.nn.functional.interpolate(torch.from_numpy(attention_now).unsqueeze(0), inputs.shape[-1], mode='bilinear')[0].mean(0).numpy()
            
            # Get approximate center of mass
            com_attn = np.copy(aggregated_attention)
            com_attn[com_attn < np.percentile(com_attn, 96)] = 0.0
            aggregated_attention[aggregated_attention < np.percentile(aggregated_attention, 80)] = 0.0
            route_steps.append(find_center_of_mass(com_attn))


            colour = list(cmap(step_linspace[stepi]))
            route_colours.append(colour)

            mapped_attention = torch.nn.functional.interpolate(torch.from_numpy(attention_now).unsqueeze(0), inputs.shape[-1], mode='bilinear')[0].mean(0).numpy()
            mapped_attention = (mapped_attention - mapped_attention.min())/np.ptp(mapped_attention)
            # np.clip(guess_maze * (1-mapped_attention[...,np.newaxis]*0.5) + (cmap_viridis(mapped_attention)[:,:,:3] * mapped_attention[...,np.newaxis])*1.3, 0, 1)
            overlay_img = np.clip(guess_maze * (1-mapped_attention[...,np.newaxis]*0.6) + (cmap_viridis(mapped_attention)[:,:,:3] * mapped_attention[...,np.newaxis])*1.1, 0, 1)#np.clip((np.copy(guess_maze)*(1-aggregated_attention[:,:,np.newaxis])*0.7 + (aggregated_attention[:,:,np.newaxis]*3 * np.reshape(np.array(colour)[:3], (1, 1, 3)))), 0, 1)
            axes['overlay'].imshow(overlay_img)

            y_coords, x_coords = zip(*route_steps)
            y_coords = inputs.shape[-1] - np.array(list(y_coords))-1

            
            axes['route'].imshow(np.flip(np.moveaxis(inputs, 0, -1), axis=0), origin='lower')
            # ax.imshow(np.flip(solution_maze, axis=0), origin='lower')
            arrow_scale = 2
            for i in range(len(route_steps)-1):
                dx = x_coords[i+1] - x_coords[i]
                dy = y_coords[i+1] - y_coords[i]
                axes['route'].arrow(x_coords[i], y_coords[i], dx, dy, linewidth=2*arrow_scale, head_width=0.2*arrow_scale, head_length=0.3*arrow_scale, fc=route_colours[i], ec=route_colours[i], length_includes_head = True)
                
            fig.tight_layout(pad=0.1) # Adjust spacing

            # Render the plot to a numpy array
            canvas = fig.canvas
            canvas.draw()
            image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3] # Get RGB

            frames.append(image_numpy) # Add to list for GIF

            # fig.savefig(f'{save_location}/frame.png', dpi=200)

            plt.close(fig) 

            # # frame = np.clip((np.copy(guess_maze)*0.5 + (aggregated_attention[:,:,np.newaxis] * np.reshape(np.array(colour)[:3], (1, 1, 3)))), 0, 1)
            # frame = torch.nn.functional.interpolate(torch.from_numpy(frame).permute(2,0,1).unsqueeze(0), 256)[0].permute(1,2,0).detach().cpu().numpy()
            # frames.append((frame*255).astype(np.uint8))
            pbar.update(1)


    y_coords, x_coords = zip(*route_steps)
    y_coords = inputs.shape[-1] - np.array(list(y_coords))-1

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    ax.imshow(np.flip(np.moveaxis(inputs, 0, -1), axis=0), origin='lower')
    # ax.imshow(np.flip(solution_maze, axis=0), origin='lower')
    arrow_scale = 2
    for i in range(len(route_steps)-1):
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[i+1] - y_coords[i]
        plt.arrow(x_coords[i], y_coords[i], dx, dy, linewidth=2*arrow_scale, head_width=0.2*arrow_scale, head_length=0.3*arrow_scale, fc=route_colours[i], ec=route_colours[i], length_includes_head = True)

    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(f'{save_location}/route_approximation.png', dpi=200)
    imageio.mimsave(f'{save_location}/prediction.gif', frames, fps=15, loop=100)
    plt.close(fig)


def visualize_ctm_dashboard(model, synch_out, save_path, step_num):
    sns.set_context("talk", font_scale=1.0)
    sns.set_style("whitegrid")
    
    activity = np.mean(synch_out, axis=0)
    decay_params = model.decay_params_out.detach().cpu().numpy()
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    is_hub_edge = (left == right)
    
    hub_edge_indices = np.where(is_hub_edge)[0] 
    hub_decays = decay_params[is_hub_edge]
    
    # SORTING: Slow -> Fast (Sort the Edge IDs based on their decay value)
    sort_idx = np.argsort(hub_decays)
    sorted_edge_indices = hub_edge_indices[sort_idx]
    
    # Data for plots
    hub_activity_T = activity[:, sorted_edge_indices].T      # (Hubs, Time)
    hub_activity_time = activity[:, sorted_edge_indices]     # (Time, Hubs)
    
    global_mean_activity = np.abs(activity).mean(axis=0)
    hubs_mean_activity = global_mean_activity[sorted_edge_indices]

    # =========================================================
    # METRICS
    # =========================================================
    dead_hub_rate = (hubs_mean_activity < 1e-6).mean()
    
    n = len(hubs_mean_activity)
    sorted_act = np.sort(hubs_mean_activity)
    
    # Gini Coefficient
    lorenz_curve = np.cumsum(sorted_act) / (sorted_act.sum() + 1e-9)
    gini = ((2 * np.arange(1, n + 1) - n - 1) * sorted_act).sum() / (n * sorted_act.sum() + 1e-9)
    
    # Entropy
    p = sorted_act / (sorted_act.sum() + 1e-9)
    entropy = -np.sum(p * np.log(p + 1e-9))
    
    # Effective Rank (SVD on Temporal Dynamics)
    try:
        _, S, _ = np.linalg.svd(hub_activity_time.T)
        norm_S = S / (S.sum() + 1e-9)
        effective_rank = np.exp(-np.sum(norm_S * np.log(norm_S + 1e-9)))
    except:
        effective_rank = 0.0
        S = np.zeros(10)

    # =========================================================
    # PLOTTING
    # =========================================================
    fig = plt.figure(figsize=(22, 18))
    
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1.2, 1], hspace=0.4, wspace=0.25)
    
    COLOR_HUB = '#E056FD'  # Neon Purple
    COLOR_CONN = '#FFAA00' # Orange
    
    # --- ROW 1: STRUCTURE (Decay Distribution) ---
    ax_struct = plt.subplot(gs[0, :])
    bins = np.linspace(-0.2, 5.5, 150)
    
    sns.histplot(decay_params[is_hub_edge], color=COLOR_HUB, element="step", fill=True, alpha=0.3, bins=bins, ax=ax_struct, label='Hubs (Core)')
    sns.histplot(decay_params[is_hub_edge], color=COLOR_HUB, element="step", fill=False, lw=3, bins=bins, ax=ax_struct)
    sns.histplot(decay_params[~is_hub_edge], color=COLOR_CONN, element="step", fill=True, alpha=0.3, bins=bins, ax=ax_struct, label='Connections')
    
    ax_struct.set_yscale('log')
    ax_struct.set_xlim(-0.2, 5.5)
    ax_struct.set_title(f"1. Structure: Decay Parameter Distribution (Step {step_num})", fontweight='bold')
    ax_struct.set_xlabel("Decay Parameter p (r = e^-p)")
    
    # Reference Zones
    ylim = ax_struct.get_ylim()
    # Zone 1: Memory (0.0 - 0.3)
    ax_struct.axvspan(0.0, 0.3, color=COLOR_HUB, alpha=0.1)
    ax_struct.text(0.15, ylim[1]*0.5, "Scale-Free\nMemory\n(r > 0.74)", 
                   ha='center', color=COLOR_HUB, fontweight='bold', fontsize=12)
    # Zone 2: Lattice (0.5)
    ax_struct.axvline(0.5, color='green', linestyle='--', alpha=0.5)
    ax_struct.text(0.55, ylim[1]*0.2, "Lattice (0.5)", color='green', fontsize=10)
    # Zone 3: Rewired/Feeders (3.0)
    ax_struct.axvline(3.0, color='cyan', linestyle='--', alpha=0.5)
    ax_struct.text(3.05, ylim[1]*0.2, "Feeders (3.0)", color='cyan', fontsize=10)
    
    ax_struct.legend(loc='upper right', frameon=True)

    # --- ROW 2A: BEHAVIOR (Sorted Heatmap) ---
    ax_behav = plt.subplot(gs[1, 0])
    robust_max = np.percentile(hub_activity_T, 99)
    sns.heatmap(hub_activity_T, ax=ax_behav, cmap="magma", cbar=False, vmin=0, vmax=robust_max)
    ax_behav.set_title("2A. Behavior (Sorted by Decay)", fontweight='bold')
    ax_behav.set_ylabel("Hubs (Slow → Fast)")
    ax_behav.set_xlabel("Time Steps")
    
    # --- ROW 2B: PHASE LOCKING (Sorted Correlation) ---
    ax_corr = plt.subplot(gs[1, 1:])
    corr = np.corrcoef(hub_activity_time.T + 1e-9) 
    sns.heatmap(corr, ax=ax_corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
    ax_corr.set_title("2B. Phase-Locking (Sorted by Decay)", fontweight='bold')
    ax_corr.set_xlabel("Hubs (Slow → Fast)")
    ax_corr.set_ylabel("Hubs (Slow → Fast)")

    # --- ROW 3A: INEQUALITY (Gini) ---
    ax_gini = plt.subplot(gs[2, 0])
    ax_gini.plot(np.linspace(0, 1, n), lorenz_curve, color=COLOR_HUB, lw=4)
    ax_gini.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax_gini.set_title(f"3A. Gini: {gini:.2f}", fontweight='bold')
    ax_gini.set_ylabel("Cumulative Activity")
    ax_gini.text(0.6, 0.1, "Winner-Take-All" if gini > 0.5 else "Democratic", transform=ax_gini.transAxes, fontsize=10)
    
    # --- ROW 3B: RANK (SVD) ---
    ax_rank = plt.subplot(gs[2, 1])
    ax_rank.plot(S[:min(20, len(S))], marker='o', color='green', markersize=8)
    ax_rank.set_title(f"3B. Rank: {effective_rank:.1f}", fontweight='bold')
    ax_rank.set_ylabel("Singular Value")

    # --- ROW 3C: LOAD (Entropy) ---
    ax_hist = plt.subplot(gs[2, 2])
    sns.histplot(hubs_mean_activity, bins=15, ax=ax_hist, color='orange', kde=True)
    ax_hist.set_title(f"3C. Entropy: {entropy:.2f}", fontweight='bold')
    ax_hist.set_xlabel("Mean Activity")
    ax_hist.text(0.95, 0.95, f"Dead: {dead_hub_rate*100:.0f}%", 
                 transform=ax_hist.transAxes, ha='right', va='top', color='red', fontweight='bold')

    plt.savefig(save_path)
    plt.close()
    
    # Reset context to avoid affecting other plots in the loop
    sns.reset_orig()

    return {"gini": gini, "entropy": entropy, "rank": effective_rank, "dead_hubs": dead_hub_rate}


def visualize_topology_matrix(model, save_path="sw_matrix.png"):
    """Visualizes the Small-World connectivity as an Adjacency Matrix Scatter Plot."""
    left = model.out_neuron_indices_left.detach().cpu().numpy()
    right = model.out_neuron_indices_right.detach().cpu().numpy()
    decays = model.decay_params_out.detach().cpu().numpy()
    
    is_self = (left == right)
    hub_indices = np.unique(left[is_self])
    hub_indices.sort()
    
    num_hubs = len(hub_indices)
    if num_hubs == 0:
        print("Warning: No Hubs found for Matrix visualization.")
        return

    # Map real ID (e.g., 40, 80) -> Logical ID (e.g., 1, 2)
    idx_map = {real_id: i for i, real_id in enumerate(hub_indices)}
    
    # Filter edges to ONLY show Core-to-Core connections (Hub -> Hub)
    # This removes Feeders to focus strictly on the Ring/Lattice structure
    mask_core = np.isin(left, hub_indices) & np.isin(right, hub_indices)
    
    core_left = left[mask_core]
    core_right = right[mask_core]
    core_decays = decays[mask_core]
    
    # Map to logical coordinates
    logical_left = np.array([idx_map[x] for x in core_left])
    logical_right = np.array([idx_map[x] for x in core_right])
    
    # Logical Self-Loops (Diagonal)
    mask_logical_self = (logical_left == logical_right)
    # Logical Lattice (Neighbors) vs Long-Range (if any)
    mask_logical_lattice = (~mask_logical_self) & (core_decays < 2.0)
    
    plt.figure(figsize=(10, 10))
    
    # Plot Lattice (Green) - Should form distinct bands parallel to diagonal
    plt.scatter(logical_left[mask_logical_lattice], logical_right[mask_logical_lattice], 
                c='limegreen', s=100, marker='s', label='Lattice (Ring)')

    # Plot Hubs (Purple) - The Diagonal
    plt.scatter(logical_left[mask_logical_self], logical_right[mask_logical_self], 
                c='#E056FD', s=150, marker='D', label='Hubs (Core)')
    
    plt.title(f"Logical Topology (Collapsed View)\nShowing {num_hubs} Hubs | Expected: Tridiagonal Band")
    plt.xlabel("Hub Logical Index (0..N)")
    plt.ylabel("Hub Logical Index (0..N)")
    
    # Set integer ticks if small enough
    if num_hubs <= 50:
        plt.xticks(np.arange(0, num_hubs, 5))
        plt.yticks(np.arange(0, num_hubs, 5))
        plt.grid(True, which='both', alpha=0.1, color='white')
    else:
        plt.grid(False)

    plt.gca().invert_yaxis()
    plt.legend(loc='upper right')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- CONFIGURATION: NEON PALETTE ---
COLOR_HUB     = "#E056FD"  # Electric Purple
COLOR_REWIRED = "#00FFFF"  # Cyan
COLOR_LATTICE = "#00FF7F"  # Spring Green
COLOR_NOISE   = "#FF0055"  # Hot Pink/Red

def get_edge_properties(s, t, decay_val, active_hubs):
    """
    Helper to classify edge types and weights based on memory horizon.

    - Param ~0.1 -> Hub (Storage / Leaky Integrator)
    - Param ~0.5 -> Lattice (Relay / Short-Term Context)
    - Param ~3.0 -> Rewired (Transmission / Impulse)
    """
    decay_val = float(decay_val)

    if s == t:
        # Hubs: Self-loops
        return "Hub (Storage)", COLOR_HUB, 1.0 
        
    elif decay_val > 2.0:
        # Fast Transmission (Param ~ 3.0)
        # We use > 2.0 to safely capture the N(3.0, 0.2) distribution
        if t in active_hubs:
            return "Rewired (Transmission)", COLOR_REWIRED, 1.0
        else:
            return "Noise", COLOR_NOISE, 0.1
            
    elif decay_val <= 2.0:
        # Fast Relay (Param ~ 0.5)
        # Visual: High weight (5.0) to force Gephi to show the Ring
        return "Lattice (Relay)", COLOR_LATTICE, 5.0
        
    return "Unknown", "#FFFFFF", 0.1

def export_full_network(model, save_path="sw_full_network.graphml"):
    """
    Exports the ENTIRE network topology.
    """
    
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    decays = model.decay_params_out.detach().cpu().numpy()
    
    # Identify Real Hubs via Self-Loops
    # Only neurons with i->i connections are state-maintaining Hubs.
    active_hubs = set(left[left == right])
    
    G = nx.DiGraph()
    
    for s, t, d in zip(left, right, decays):
        s, t = int(s), int(t)
        category, color, weight = get_edge_properties(s, t, d, active_hubs)
        
        # Source Node 's'
        if s in active_hubs:
            cat_s, col_s, size_s = "Hub", COLOR_HUB, 30.0
        else:
            # If s is not in active_hubs, it is a Sensory Input (Periphery)
            cat_s, col_s, size_s = "Sensory", "#AAAAAA", 10.0 # Grey for raw input
        G.add_node(s, label=f"Neuron {s}", Category=cat_s, Size=size_s, Color=col_s)

        # Target Node 't'
        if not G.has_node(t):
            if t in active_hubs:
                cat_t, col_t, size_t = "Hub", COLOR_HUB, 30.0
            else:
                cat_t, col_t, size_t = "Neighbor", COLOR_LATTICE, 10.0
            G.add_node(t, label=f"Neuron {t}", Category=cat_t, Size=size_t, Color=col_t)

        G.add_edge(s, t, weight=weight, Type=category, Color=color, Param=float(d))
        
    nx.write_graphml(G, save_path)
