
import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
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

    # --- Wiggle Index Calculation ---
    steps_arr = np.array(route_steps)
    if len(steps_arr) < 3:
        return 0.0

    window = 10 # Check for panic in 10-step bursts
    if len(steps_arr) < window: window = len(steps_arr)
    
    max_wiggle = 0.0
    for i in range(len(steps_arr) - window + 1):
        segment = steps_arr[i : i+window]
        
        # Path Length in this window
        seg_path = np.sum(np.linalg.norm(segment[1:] - segment[:-1], axis=1))
        
        # Net Displacement in this window
        seg_disp = np.linalg.norm(segment[-1] - segment[0])
        
        if seg_path < 1e-6:
            val = 0.0
        else:
            val = (seg_path - seg_disp) / seg_path

        if val > max_wiggle:
            max_wiggle = val
            
    return max_wiggle


# --- CONFIGURATION ---
COLOR_CORE      = "#E056FD"  # Electric Purple (Memory)
COLOR_PERIPHERY = "#00FFFF"  # Cyan (Sensory)

def get_edge_properties(s, t, decay_val, active_hubs):
    """Classifies edges for GraphML export."""
    decay_val = float(decay_val)

    if s in active_hubs and t in active_hubs:
        if s == t:
            return "Core (Memory)", COLOR_CORE, 2.0 
        else:
            return "Core (Ring)", "#FFD700", 2.0

    return "Periphery (Sensory)", COLOR_PERIPHERY, 0.5

def visualize_ctm_dashboard(model, activity_time_mean, vitals, save_prefix, step_num):
    """
    Unified Dashboard for CTM Health.
    
    Args:
        model: The CTM model instance.
        activity_time_mean: Pre-computed mean activity (Time, Edges).
        vitals: Dict containing pre-calculated metrics (Rank, Dead, etc.).
        save_prefix: Path prefix.
        step_num: Current iteration.
    """
    decay_params = model.decay_params_out.detach().cpu().numpy()
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    self_loop_mask = (left == right)

    # Rely on the fact that 'left' is constructed as [Hubs(Self), Hubs(Ring), Feeders].
    # This extracts the Hub IDs in their TOPOLOGICAL order (Ring Order), not numerical.
    hub_ids = left[self_loop_mask] 
    
    is_src_hub = np.isin(left, hub_ids)
    is_tgt_hub = np.isin(right, hub_ids)
    mask_core = is_src_hub & is_tgt_hub   # Includes Self-Loops, Ring, AND Shortcuts
    mask_ring = mask_core & (~self_loop_mask)
    mask_peri = (~mask_core)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    plt.suptitle(f"CTM Dashboard | Step {step_num}", fontsize=16, fontweight='bold')
    
    # =========================================================
    # PLOT 1 (Top-Left): Dynamics Distribution (Physics Check)
    # =========================================================
    ax = axes[0, 0]
    bins = np.linspace(-0.2, 4, 100)
    
    sns.histplot(decay_params[mask_core], color="magenta", label="Core (Mem + Lateral)", 
                 element="step", fill=True, alpha=0.5, bins=bins, ax=ax)
    sns.histplot(decay_params[mask_peri], color="cyan", label="Periphery (Sensory)", 
                 element="step", fill=True, alpha=0.3, bins=bins, ax=ax)
    
    ax.set_yscale('log')
    ax.set_title("Dynamics Segregation (Decay Params)")
    ax.set_xlabel("Parameter p")
    ax.legend()
    # Reference Zones
    ax.axvspan(0, 0.4, color='magenta', alpha=0.1) # Safe Core
    ax.axvspan(0.5, 2.5, color='cyan', alpha=0.1)  # Safe Periphery

    # =========================================================
    # PLOT 2 (Top-Mid): Singular Value Spectrum (Rank Check)
    # =========================================================
    ax = axes[0, 1]
    # We use the SVD values passed from train.py to ensure consistency
    if 'singular_values' in vitals:
        S = vitals['singular_values']
        ax.plot(S[:20], marker='o', color='green', lw=2)
    ax.set_title(f"Dimensionality (Rank: {vitals['rank']:.1f})")
    ax.set_xlabel("Component")
    ax.grid(True, alpha=0.3)

    # =========================================================
    # PLOT 3 (Top-Right): Hub Connectivity (The Switchboard Check)
    # =========================================================
    ax = axes[0, 2]
    steering_ratio = 0.0
    if mask_core.any():
        n_hubs = len(hub_ids)
        id_map = {global_id: i for i, global_id in enumerate(hub_ids)}
        hub_matrix = np.zeros((n_hubs, n_hubs))
        mean_edge_act = activity_time_mean.mean(axis=0)
        
        core_indices = np.where(mask_core)[0]
        for idx in core_indices:
            u, v = left[idx], right[idx]
            hub_matrix[id_map[u], id_map[v]] = mean_edge_act[idx]

        #  Ring Edges: (i, i+1) in sorted matrix
        ring_energy = 0.0
        for i in range(n_hubs):
            target = (i + 1) % n_hubs
            ring_energy += hub_matrix[i, target]

        # Self Edges: (i, i)
        self_energy = np.trace(hub_matrix)
        
        # Shortcut Edges: Total - Ring - Self
        total_energy = np.sum(hub_matrix)
        shortcut_energy = total_energy - ring_energy - self_energy
        if ring_energy > 1e-9:
            steering_ratio = shortcut_energy / ring_energy
            
        sns.heatmap(hub_matrix, cmap="magma", ax=ax, cbar=True, square=True, xticklabels=False, yticklabels=False)
        
    ax.set_title(f"Hub Matrix (Lateral Strength: {vitals.get('ring_strength', 0):.2f} | Steering Ratio: {steering_ratio:.2f})")
    ax.set_xlabel("Target Hub")
    ax.set_ylabel("Source Hub")

    # =========================================================
    # PLOT 4 (Mid-Left): Core Memory Trace
    # =========================================================
    ax = axes[1, 0]
    if self_loop_mask.any():
        sns.heatmap(activity_time_mean[:, self_loop_mask].T, cmap="magma", ax=ax, cbar=False,
                    yticklabels=False)
    ax.set_ylabel(f"Hubs ({self_loop_mask.sum()})")
    ax.set_title("Memory Registers (Self-Loops Only)")
    ax.set_xlabel("Internal Ticks")

    # =========================================================
    # PLOT 5 (Bot-Mid): Load Balance (Lorenz Curve)
    # =========================================================
    ax = axes[1, 1]
    if self_loop_mask.any():
        total_act = activity_time_mean[:, self_loop_mask].sum(axis=0)
        sorted_act = np.sort(total_act)
        lorenz = np.cumsum(sorted_act) / (sorted_act.sum() + 1e-9)
        ax.plot(np.linspace(0, 1, len(lorenz)), lorenz, color='purple', lw=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_title(f"Load Balance (Dead: {vitals['dead']:.1f}%)")
    ax.set_xlabel("Cumulative % of Hubs")
    ax.set_ylabel("Cumulative % of Activity")

    # =========================================================
    # PLOT 6 (Bot-Right): Feeder Activity (Sensory Input)
    # =========================================================
    ax = axes[1, 2]
    if mask_peri.any():
        peri_act = activity_time_mean[:, mask_peri].T
        # Sort by intensity
        sort_idx = np.argsort(peri_act.sum(axis=1))
        # Downsample if too many feeders
        if len(sort_idx) > 100: sort_idx = sort_idx[::len(sort_idx)//100]
        sns.heatmap(peri_act[sort_idx], cmap="viridis", ax=ax, cbar=False, yticklabels=False)
    ax.set_title("Sensory Feeder Activity")
    ax.set_ylabel("Feeder Connections")
    ax.set_xlabel("Internal Ticks")

    # =========================================================
    # PLOT 7 (Bot-Left): Hydraulic Lag (Cross-Correlation)
    # =========================================================
    ax = axes[2, 0]
    if self_loop_mask.any() and mask_peri.any():
        # Extract Traces (Mean Activity over Time)
        hub_trace = activity_time_mean[:, self_loop_mask].mean(axis=1) # (T,)
        peri_trace = activity_time_mean[:, mask_peri].mean(axis=1)     # (T,)
        
        # Normalize (Z-Score) for Correlation
        hub_trace = (hub_trace - hub_trace.mean()) / (hub_trace.std() + 1e-9)
        peri_trace = (peri_trace - peri_trace.mean()) / (peri_trace.std() + 1e-9)
        
        # Cross Correlate (Full Mode)
        xcorr = np.correlate(hub_trace, peri_trace, mode='full') / len(hub_trace)
        lags = np.arange(1 - len(hub_trace), len(hub_trace))
        
        # Find Peak
        max_idx = np.argmax(xcorr)
        max_lag = lags[max_idx]

        ax.plot(lags, xcorr, color='orange', lw=2)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.plot(max_lag, xcorr[max_idx], 'ro') # Mark peak with red dot
        ax.set_title(f"Hydraulic Lag (Hub vs Feeder)\nPeak Lag: {max_lag}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")
        ax.grid(True, alpha=0.3)

    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_dashboard_{step_num}.png")
    plt.close()

def export_full_network(model, save_path="sw_full_network.graphml"):
    """Exports topology to GraphML."""
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()
    decays = model.decay_params_out.detach().cpu().numpy()
    
    active_hubs = set(left[left == right])
    
    G = nx.DiGraph()
    for s, t, d in zip(left, right, decays):
        s, t = int(s), int(t)
        cat, col, weight = get_edge_properties(s, t, d, active_hubs)
        
        for n in (s, t):
            if not G.has_node(n):
                is_hub = n in active_hubs
                G.add_node(n, label=f"N{n}", Category="Hub" if is_hub else "Sensory", 
                           Color=COLOR_CORE if is_hub else "#AAAAAA", Size=30.0 if is_hub else 10.0)
                       
        G.add_edge(s, t, weight=weight, Type=cat, Color=col, Param=float(d))
        
    nx.write_graphml(G, save_path)


def visualize_topology_matrix(model, save_path="sw_matrix.png"):
    """Visualizes the connectivity matrix with Ring highlighting."""
    left = model.out_neuron_indices_left.cpu().numpy()
    right = model.out_neuron_indices_right.cpu().numpy()

    self_loop_mask = (left == right)
    hub_ids = np.unique(left[self_loop_mask])
    
    is_src_hub = np.isin(left, hub_ids)
    is_tgt_hub = np.isin(right, hub_ids)
    
    mask_self = self_loop_mask
    mask_ring = (is_src_hub & is_tgt_hub) & (~self_loop_mask)
    mask_peri = (~is_src_hub) # Feeders
    
    plt.figure(figsize=(10, 10))
    
    # Layer 1: Periphery (Blue background noise)
    plt.scatter(left[mask_peri], right[mask_peri], 
                c='dodgerblue', s=15, alpha=0.3, label='Sensory', marker='.')

    # Layer 2: Ring Connections (Gold connectors)
    plt.scatter(left[mask_ring], right[mask_ring], 
                c='gold', s=40, alpha=1.0, label='Ring', marker='o', edgecolors='k', linewidth=0.5)

    # Layer 3: Memory Registers (Magenta Anchors)
    plt.scatter(left[mask_self], right[mask_self], 
                c='magenta', s=80, alpha=1.0, label='Hub Memory', marker='D', edgecolors='white')

    num_hubs = len(hub_ids)
    plt.title(f"Hub & Spoke Topology (Hubs={num_hubs})")
    plt.xlabel("Source Neuron")
    plt.ylabel("Target Neuron")
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

