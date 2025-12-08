
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
    
    Note: decay_param and r are related by r = exp(-decay_param)
    High decay_param → Low r → Long memory
    
    Args:
        model: CTM model (for indices/params)
        synch_out_viz: NumPy array (Iterations, Batch, Pairs) from the test batch
        save_prefix: Path prefix
        step_num: Iteration number
    """
    decay_params = model.decay_params_out.detach().cpu().numpy()
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    is_hub = (left == right)
    
    # =========================================================
    # PLOT 1: Topology Structure (Log-Scale Histogram)
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Hubs (Blue) - should peak near 15
    sns.histplot(decay_params[is_hub], color="blue", label="Hubs (Self-Pairs)", 
                 kde=False, bins=60, alpha=0.6, ax=ax)
    # Plot Non-Hubs (Orange) - bimodal: lattice (~2.3) and rewired (~15)
    sns.histplot(decay_params[~is_hub], color="orange", label="Connections", 
                 kde=False, bins=60, alpha=0.6, ax=ax)
    
    ax.set_yscale('log')
    ax.set_xlim(-0.5, 15.5)
    
    # Reference Lines & Labels
    ylim = ax.get_ylim()
    text_y = 10**(np.log10(ylim[1]) * 0.85)
    
    # Label the three regions based on decay_param values
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=2.3, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=10.0, color='gray', linestyle='--', alpha=0.3)
    
    ax.text(0.25, text_y, "Fast Decay\nr≈1\n(Noise)", 
            ha='center', va='bottom', color='gray', fontsize=9, fontweight='bold')
    ax.text(1.4, text_y, "Working\nr≈0.1\n(Lattice)", 
            ha='center', va='bottom', color='gray', fontsize=9, fontweight='bold')
    ax.text(12.5, text_y, "Infinite\nr≈0\n(Global)", 
            ha='center', va='bottom', color='blue', fontsize=9, fontweight='bold')
    
    ax.set_title(f"Small-World Topology at Step {step_num}\n(Decay Parameter Distribution)")
    ax.set_xlabel("Decay Parameter (higher → longer memory)")
    ax.set_ylabel("Count (Log Scale)")
    ax.legend(loc='upper left')
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
    
    # Sort non-hubs by decay_param (Low/Noise -> High/Infinite)
    sort_indices = np.argsort(nonhub_decays)
    nonhub_activity_sorted = nonhub_activity[sort_indices, :]
    
    # Downsample if needed (to match visual height of Hubs for comparison)
    # We use nearest-neighbor interpolation via linspace
    if nonhub_activity_sorted.shape[0] > hubs_activity.shape[0] * 2:
        indices = np.linspace(0, nonhub_activity_sorted.shape[0]-1, 
                             hubs_activity.shape[0] * 2, dtype=int)
        nonhub_activity_sorted = nonhub_activity_sorted[indices, :]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Use robust quantiles for vmin/vmax to prevent outliers from washing out the plot
    # But strictly 0 minimum since activity is positive (ReLU/GLU/etc output usually)
    robust_max = np.percentile(activity_avg, 99) 
    
    sns.heatmap(hubs_activity, ax=ax1, cmap="magma", cbar=True, vmin=0, vmax=robust_max)
    ax1.set_title("Hub Activity (Self-Pairs, Infinite Memory)")
    ax1.set_ylabel("Hub Neurons")
    
    sns.heatmap(nonhub_activity_sorted, ax=ax2, cmap="viridis", cbar=True, vmin=0, vmax=robust_max)
    ax2.set_title("Connection Activity (Sorted by Decay Parameter)\n" + 
                 "(Bottom = Fast Decay, Top = Infinite)")
    ax2.set_ylabel("Connections\n(Sorted Low $\\to$ High Decay)")
    ax2.set_xlabel("Time Steps")
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_behavior_{step_num}.png")
    plt.close()


def visualize_evolution_metrics(model, synch_out_viz, save_path="sw_evolution.png"):
    activity = synch_out_viz.detach().cpu()
    
    # 1. Hub Extraction (Safe Mode)
    left = model.out_neuron_indices_left.detach().cpu().numpy()
    right = model.out_neuron_indices_right.detach().cpu().numpy()
    self_loop_edge_mask = (left == right)
    
    # Calculate global mean activity over Batch and Time (Shape: [Neurons])
    global_mean_activity = activity.abs().mean(dim=(0, 1)).numpy() 
    # Use a flag to track if we found hubs or fell back to global
    hubs_only_flag = False 
    if self_loop_edge_mask.any():
        # Get the actual Neuron IDs that are hubs
        hub_neuron_ids = left[self_loop_edge_mask]
        # Slice the global_mean_activity vector using the correct IDs
        hubs_only = global_mean_activity[hub_neuron_ids]
        hubs_only_flag = True
    else:
        # Fallback: If no self-loops, treat top 10% most active neurons as "functional hubs"
        threshold = np.percentile(global_mean_activity, 90)
        hubs_only = global_mean_activity[global_mean_activity > threshold]

    # 2. Vital Signs
    dead_hub_rate = (hubs_only < 1e-6).sum() / (len(hubs_only) + 1e-9)
    global_energy = global_mean_activity.sum() # Global energy across all neurons

    # 3. Distribution Metrics
    # ---------------------------------------------------------
    # Gini (Inequality)
    sorted_hubs = np.sort(hubs_only)
    n = len(hubs_only)
    index = np.arange(1, n + 1)
    # Add epsilon to divisor to prevent div/0
    gini = ((2 * index - n - 1) * sorted_hubs).sum() / (n * sorted_hubs.sum() + 1e-9)

    # Entropy (Sharpness)
    p = hubs_only / (hubs_only.sum() + 1e-9)
    entropy = -np.sum(p * np.log(p + 1e-9))

    # Effective Rank (Diversity)
    # We measure the SVD of the activity, averaged over time (axis 1)
    batch_activity = activity.abs().mean(dim=1).numpy()
    try:
        _, S, _ = np.linalg.svd(batch_activity)
        singular_vals = S / S.sum()
        effective_rank = np.exp(-np.sum(singular_vals * np.log(singular_vals + 1e-9)))
    except:
        effective_rank = 0.0 # Handle singular matrix crash

    # 4. Visualization
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
    ax[2].set_title(f"Feature Diversity\nRank: {effective_rank:.1f}/{min(activity.shape[0], activity.shape[2])}")
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
    """
    Visualizes the Small-World connectivity as an Adjacency Matrix Scatter Plot.
    This is often superior to circular graph plots for verifying locality.
    """
    # 1. Extract Data
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    decays = model.decay_params_out.detach().cpu().numpy()
    
    d_model = model.d_model
    
    # 2. Classify Edges
    # Masks based on your confirmed logic
    mask_self = (left == right)
    mask_noise = (~mask_self) & (decays < 1.0)
    mask_lattice = (~mask_self) & (decays > 1.0) & (decays < 10.0)
    mask_rewired = (~mask_self) & (decays > 10.0)
    
    # 3. Setup Plot
    plt.figure(figsize=(12, 12))
    
    # A. Plot Rewired (Blue) - Global Scatter
    # Plot these first so they are in the background
    plt.scatter(left[mask_rewired], right[mask_rewired], 
                c='blue', s=15, alpha=0.4, label='Rewired (Global Shortcut)', marker='.')
                
    # B. Plot Lattice (Green) - The Diagonal Band
    # Plot these second to ensure they are visible
    plt.scatter(left[mask_lattice], right[mask_lattice], 
                c='green', s=15, alpha=0.6, label='Lattice (Working Memory)', marker='.')

    # C. Plot Hubs (Dark Blue) - The Spine
    plt.scatter(left[mask_self], right[mask_self], 
                c='navy', s=40, alpha=1.0, label='Hubs (Infinite Memory)', marker='o')

    # D. Plot Noise (Red)
    plt.scatter(left[mask_noise], right[mask_noise], 
                c='red', s=20, alpha=1.0, label='Noise (Fast Decay)', marker='x')

    # 4. Styling
    plt.title(f"Small-World Adjacency Matrix\n(d_model={d_model}, Total Edges={len(left)})")
    plt.xlabel("Source Neuron Index")
    plt.ylabel("Target Neuron Index")
    plt.xlim(0, d_model)
    plt.ylim(0, d_model)
    
    # Invert Y axis to match standard matrix notation (0,0 at top left)
    plt.gca().invert_yaxis()
    
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# --- CONFIGURATION: NEON PALETTE (Dark Mode Optimized) ---
# Designed to pop against Black/Dark Grey backgrounds
COLOR_HUB     = "#E056FD"  # Electric Purple (The Anchors)
COLOR_REWIRED = "#00FFFF"  # Cyan (The Shortcuts)
COLOR_LATTICE = "#00FF7F"  # Spring Green (The Mesh)
COLOR_NOISE   = "#FF0055"  # Hot Pink/Red (The Noise)

def get_edge_properties(s, t, decay_val):
    """
    Helper to classify edge types and weights based on memory horizon.
    Returns: (Category, Color, Weight)
    """
    # Weight Scaling: Maps Decay Param to Physics Gravity
    # Param 15.0 -> Weight ~3.0 (Strong Pull)
    # Param  2.3 -> Weight ~0.4 (Weak Pull)
    SCALE_FACTOR = 0.2
    
    raw_weight = float(decay_val)
    viz_weight = max(raw_weight * SCALE_FACTOR, 0.1)

    if s == t:
        return "Hub (Infinite)", COLOR_HUB, 4.0 # Force Hubs to be very rigid
    elif decay_val > 10.0:
        return "Rewired (Global)", COLOR_REWIRED, viz_weight
    elif decay_val > 1.0:
        return "Lattice (Working)", COLOR_LATTICE, viz_weight
    else:
        return "Noise (Fast)", COLOR_NOISE, 0.1 # Noise has almost no gravity

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
        
        # Node Sizing Logic
        # Active Hubs (Sources) are Large (20)
        # Passive Targets are Small (10) unless they are also Hubs
        s_size = 20.0
        t_size = 20.0 if t in active_hubs else 8.0
        
        G.add_node(s, label=f"Neuron {s}", Category="Hub", Size=s_size, Color=COLOR_HUB)
        if not G.has_node(t):
            G.add_node(t, label=f"Neuron {t}", Category="Neighbor", Size=t_size, Color=COLOR_LATTICE)
            
        G.add_edge(s, t, weight=weight, Type=category, Color=color, Param=float(d))
        
    nx.write_graphml(G, save_path)


def export_to_gephi(model, save_path="sw_network.graphml", n_hubs=32):
    """
    Exports a clean, filtered subset of the Small-World topology to GraphML.
    
    Optimized for Dark Mode / Gephi:
    - Uses 'Decay Parameter' as Edge Weight (Physics: Gravity).
    - Uses 'Visual Hierarchy' for Colors (Lattice is dim to reduce clutter).
    """
    
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    decays = model.decay_params_out.detach().cpu().numpy()
    
    # Filter Logic
    unique_hubs = np.unique(left)
    selected_hubs = unique_hubs[:n_hubs] # Contiguous slice to preserve ring geometry
    mask = np.isin(left, selected_hubs)
    
    sources = left[mask]
    targets = right[mask]
    edge_decays = decays[mask]
    
    G = nx.DiGraph()
    
    for s, t, d in zip(sources, targets, edge_decays):
        s, t = int(s), int(t)
        category, color, weight = get_edge_properties(s, t, d)
        
        # Node Sizing Logic
        # Source is always a selected Hub -> Large
        # Target is Large ONLY if it is also in our selected subset
        t_is_hub = t in selected_hubs
        t_size = 20.0 if t_is_hub else 8.0
        t_cat = "Hub" if t_is_hub else "Neighbor"
        
        G.add_node(s, label=f"Neuron {s}", Category="Hub", Size=20.0, Color=COLOR_HUB)
        
        # Only add target if not present (preserve Hub properties if it exists)
        if not G.has_node(t):
            G.add_node(t, label=f"Neuron {t}", Category=t_cat, Size=t_size, Color=COLOR_LATTICE if not t_is_hub else COLOR_HUB)
            
        G.add_edge(s, t, weight=weight, Type=category, Color=color, Param=float(d))

    nx.write_graphml(G, save_path)
