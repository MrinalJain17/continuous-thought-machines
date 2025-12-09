
import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def visualize_small_world_diagnostics(model, synch_out_viz, save_prefix, step_num):
    """
    Generates diagnostics for Small-World CTM.
    1. Histogram: Shows distribution of decay_param values (NOT decay rates r)
    2. Heatmap: Sorted by decay_param to visualize functional hierarchy

    r = exp(-decay_param)
    - param ≈ 0.0  → r ≈ 1.0 (Infinite Memory / Hubs)
    - param ≈ 0.1  → r ≈ 0.9 (Working Memory / Lattice)
    - param ≈ 15.0 → r ≈ 0.0 (Zero Memory / Rewired)
    """
    decay_params = model.decay_params_out.detach().cpu().numpy()
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    is_hub = (left == right)
    
    # =========================================================
    # PLOT 1: Topology Structure (Log-Scale Histogram)
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Hubs (Blue) - should peak near 0.0 (Infinite Memory)
    sns.histplot(decay_params[is_hub], color="blue", label="Hubs (Self-Pairs)", 
                 kde=False, bins=60, alpha=0.6, ax=ax)
    
    # Plot Non-Hubs (Orange) - bimodal: Lattice (~0.1) and Rewired (~15.0)
    sns.histplot(decay_params[~is_hub], color="orange", label="Connections", 
                 kde=False, bins=60, alpha=0.6, ax=ax)
    
    ax.set_yscale('log')
    ax.set_xlim(-0.5, 16.5)
    
    # Reference Lines & Labels
    ylim = ax.get_ylim()
    text_y = 10**(np.log10(ylim[1]) * 0.85)
    
    # Zone 1: Infinite Memory (Param ~ 0)
    ax.axvline(x=0.05, color='blue', linestyle='--', alpha=0.3)
    ax.text(0.1, text_y, "Infinite Memory\nr≈1.0\n(Hubs)", 
            ha='left', va='top', color='blue', fontsize=9, fontweight='bold')

    # Zone 2: Working Memory (Param ~ 0.1)
    ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.3)
    ax.text(0.5, text_y*0.5, "Working Memory\nr≈0.9\n(Lattice)", 
            ha='left', va='top', color='green', fontsize=9, fontweight='bold')

    # Zone 3: Zero Memory (Param ~ 15)
    ax.axvline(x=14.0, color='red', linestyle='--', alpha=0.3)
    ax.text(14.5, text_y, "Zero Memory\nr≈0.0\n(Rewired)", 
            ha='right', va='top', color='red', fontsize=9, fontweight='bold')
    
    ax.set_title(f"Small-World Topology at Step {step_num}\n(Decay Parameter Distribution)")
    ax.set_xlabel("Decay Parameter p (r = e^-p)")
    ax.set_ylabel("Count (Log Scale)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_structure_{step_num}.png")
    plt.close()

    # =========================================================
    # PLOT 2: Neural Behavior (Sorted Heatmap)
    # =========================================================
    activity_avg = synch_out_viz.mean(axis=1)  # (Iterations, Pairs)
    
    # Separate hub and non-hub activity
    hubs_activity = activity_avg[:, is_hub].T  # (num_hubs, Time)
    nonhub_activity = activity_avg[:, ~is_hub].T  # (num_nonhubs, Time)
    nonhub_decays = decay_params[~is_hub]
    
    # Sort non-hubs by decay_param (Low -> High)
    # Low Param = Infinite Memory (Top of list)
    # High Param = Zero Memory (Bottom of list)
    sort_indices = np.argsort(nonhub_decays)
    nonhub_activity_sorted = nonhub_activity[sort_indices, :]
    
    # Downsample visual
    if nonhub_activity_sorted.shape[0] > hubs_activity.shape[0] * 2:
        indices = np.linspace(0, nonhub_activity_sorted.shape[0]-1, 
                             hubs_activity.shape[0] * 2, dtype=int)
        nonhub_activity_sorted = nonhub_activity_sorted[indices, :]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    robust_max = np.percentile(activity_avg, 99) 
    
    sns.heatmap(hubs_activity, ax=ax1, cmap="magma", cbar=True, vmin=0, vmax=robust_max)
    ax1.set_title("Hub Activity (Self-Pairs)")
    ax1.set_ylabel("Hub Neurons")
    
    sns.heatmap(nonhub_activity_sorted, ax=ax2, cmap="viridis", cbar=True, vmin=0, vmax=robust_max)
    ax2.set_title("Connection Activity\n(Top = Infinite Memory/Lattice, Bottom = Zero Memory/Rewired)")
    ax2.set_ylabel("Connections (Sorted by Param)")
    ax2.set_xlabel("Time Steps")
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_behavior_{step_num}.png")
    plt.close()


def visualize_evolution_metrics(model, synch_out_viz, save_path="sw_evolution.png"):
    activity = synch_out_viz
    
    # Hub Extraction
    left = model.out_neuron_indices_left.detach().cpu().numpy()
    right = model.out_neuron_indices_right.detach().cpu().numpy()
    self_loop_edge_mask = (left == right)
    
    # Calculate global mean activity over Batch (0) and Time (1) -> Shape: [Neurons]
    global_mean_activity = np.abs(activity).mean(axis=(0, 1))
    
    hubs_only_flag = False 
    if self_loop_edge_mask.any():
        hub_neuron_ids = left[self_loop_edge_mask]
        hubs_only = global_mean_activity[hub_neuron_ids]
        hubs_only_flag = True
    else:
        # Fallback: If no self-loops, treat top 10% most active neurons as "functional hubs"
        threshold = np.percentile(global_mean_activity, 90)
        hubs_only = global_mean_activity[global_mean_activity > threshold]

    # Vital Signs
    dead_hub_rate = (hubs_only < 1e-6).sum() / (len(hubs_only) + 1e-9)
    global_energy = global_mean_activity.sum() 

    # Metrics
    sorted_hubs = np.sort(hubs_only)
    n = len(hubs_only)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_hubs).sum() / (n * sorted_hubs.sum() + 1e-9)

    p = hubs_only / (hubs_only.sum() + 1e-9)
    entropy = -np.sum(p * np.log(p + 1e-9))

    # Effective Rank
    # We measure the SVD of the activity, averaged over time (axis 1)
    batch_activity = np.abs(activity).mean(axis=1)
    try:
        _, S, _ = np.linalg.svd(batch_activity)
        singular_vals = S / S.sum()
        effective_rank = np.exp(-np.sum(singular_vals * np.log(singular_vals + 1e-9)))
    except:
        effective_rank = 0.0
        singular_vals = np.zeros(10)

    # Visualization
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    # Plot A: Lorenz Curve
    lorenz_curve = np.cumsum(sorted_hubs) / (sorted_hubs.sum() + 1e-9)
    ax[0].plot(np.linspace(0, 1, n), lorenz_curve, color='purple', lw=2)
    ax[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    title_label = "Hub" if hubs_only_flag else "Functional"
    ax[0].set_title(f"{title_label} Inequality (Gini: {gini:.2f})\nHigh = Winner-Take-All")

    # Plot B: Activity Dist (Entropy)
    sns.histplot(hubs_only, bins=20, ax=ax[1], color='orange', kde=True)
    ax[1].set_title(f"{title_label} Load (Entropy: {entropy:.2f})\nDead Hubs: {dead_hub_rate*100:.1f}%")
    ax[1].set_xlabel("Activity Magnitude")

    # Plot C: Dimensionality (Rank)
    ax[2].plot(singular_vals[:min(50, len(singular_vals))], marker='o', markersize=3, color='green')
    ax[2].set_title(f"Feature Diversity\nRank: {effective_rank:.1f}")
    ax[2].set_ylabel("Singular Val Strength")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {
        "gini": gini, 
        "entropy": entropy, 
        "rank": effective_rank, 
        "dead_hubs": dead_hub_rate,
        "energy": global_energy
    }


def visualize_topology_matrix(model, save_path="sw_matrix.png"):
    """Visualizes the Small-World connectivity as an Adjacency Matrix Scatter Plot."""
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    decays = model.decay_params_out.detach().cpu().numpy()
    d_model = model.d_model
    
    # Masks
    mask_self = (left == right)
    
    # Lattice = Working Memory (Param ~ 0.1)
    # Filter: Not Self AND Param is Small (< 1.0)
    mask_lattice = (~mask_self) & (decays < 1.0)
    
    # Rewired/Noise = Zero Memory (Param ~ 15.0)
    # Filter: Not Self AND Param is Large (> 10.0)
    mask_rewired = (~mask_self) & (decays > 10.0)
    
    plt.figure(figsize=(12, 12))
    
    # Plot Rewired (Red/Blue) - Background
    plt.scatter(left[mask_rewired], right[mask_rewired], 
                c='blue', s=15, alpha=0.3, label='Rewired/Noise (Zero Mem)', marker='.')
                
    # Plot Lattice (Green) - Foreground Band
    plt.scatter(left[mask_lattice], right[mask_lattice], 
                c='green', s=20, alpha=0.8, label='Lattice (Working Mem)', marker='.')

    # Plot Hubs (Dark Blue) - Spine
    plt.scatter(left[mask_self], right[mask_self], 
                c='navy', s=50, alpha=1.0, label='Hubs (Infinite Mem)', marker='o')

    plt.title(f"Small-World Adjacency Matrix\n(d_model={d_model}, Edges={len(left)})")
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.xlim(0, d_model)
    plt.ylim(0, d_model)
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- CONFIGURATION: NEON PALETTE ---
COLOR_HUB     = "#E056FD"  # Electric Purple
COLOR_REWIRED = "#00FFFF"  # Cyan
COLOR_LATTICE = "#00FF7F"  # Spring Green
COLOR_NOISE   = "#FF0055"  # Hot Pink/Red

def get_edge_properties(s, t, decay_val):
    """
    Helper to classify edge types and weights based on memory horizon.

    - Param 0.0  -> Hub (Infinite Mem)
    - Param 0.1  -> Lattice (Working Mem)
    - Param 15.0 -> Rewired (Zero Mem)
    
    GEPHI PHYSICS (ForceAtlas2):
    - Weight determines attraction strength.
    - We want the LATTICE (Ring) to define the shape -> High Weight.
    - We want REWIRED to be visible shortcuts but not crush the ring -> Lower Weight.
    """
    decay_val = float(decay_val)

    if s == t:
        # Hubs: Self-loops don't affect layout, but we mark them clearly
        return "Hub (Infinite)", COLOR_HUB, 1.0 
        
    elif decay_val > 10.0:
        # Param ~15.0 -> Rewired (Zero Memory)
        # Visual: Lighter weight (1.0) so they appear as "chords" across the ring
        return "Rewired (Global)", COLOR_REWIRED, 1.0
        
    elif decay_val < 2.0:
        # Param ~0.1 -> Lattice (Working Memory)
        # Visual: High weight (5.0) to force the nodes into the Ring topology
        return "Lattice (Working)", COLOR_LATTICE, 5.0
        
    else:
        # Fallback for noise or transition states
        return "Noise", COLOR_NOISE, 0.1

def export_full_network(model, save_path="sw_full_network.graphml"):
    """
    Exports the ENTIRE network topology.
    """
    
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    decays = model.decay_params_out.detach().cpu().numpy()
    
    # Identify Active Hubs (Neurons that have outgoing connections)
    active_hubs = set(np.unique(left))
    
    G = nx.DiGraph()
    
    for s, t, d in zip(left, right, decays):
        s, t = int(s), int(t)
        category, color, weight = get_edge_properties(s, t, d)
        
        # Sizing: Hubs are large anchors
        s_size = 30.0
        t_size = 30.0 if t in active_hubs else 10.0
        
        G.add_node(s, label=f"Neuron {s}", Category="Hub", Size=s_size, Color=COLOR_HUB)
        if not G.has_node(t):
            # Target might not be a source (Hub), so we label it Neighbor
            cat = "Hub" if t in active_hubs else "Neighbor"
            col = COLOR_HUB if t in active_hubs else COLOR_LATTICE
            G.add_node(t, label=f"Neuron {t}", Category=cat, Size=t_size, Color=col)
            
        G.add_edge(s, t, weight=weight, Type=category, Color=color, Param=float(d))
        
    nx.write_graphml(G, save_path)
