import argparse
import os
import random

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    # For faster
    torch.set_float32_matmul_precision('high')

from data.custom_datasets import MazeImageFolder
from models.ctm import ContinuousThoughtMachine
from models.lstm import LSTMBaseline
from models.ff import FFBaseline
from tasks.mazes.plotting import make_maze_gif, visualize_ctm_dashboard, visualize_topology_matrix, export_full_network
from tasks.image_classification.plotting import plot_neural_dynamics
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import maze_loss 
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

import torchvision

torchvision.disable_beta_transforms_warning()
console = Console()

import warnings
warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")
warnings.filterwarnings('ignore', message='divide by zero encountered in power', category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    "Corrupt EXIF data",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Metadata Warning",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Truncated File Read",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)


def make_dashboard(iter_num, total_iters, loss, acc, grad_stats, vitals_stats, optim_stats, task_stats, pbar):
    """Constructs the Live Dashboard"""
    # Main Status Table
    status_table = Table.grid(padding=(0, 2))
    status_table.add_column(style="bold cyan", justify="right")
    status_table.add_column(style="magenta")
    
    # Core Training
    status_table.add_row("Iteration", f"{iter_num}/{total_iters}")
    status_table.add_row("Loss", f"{loss:.4f}")
    status_table.add_row("Accuracy", f"{acc:.4f}")
    status_table.add_row("LR", f"{optim_stats.get('lr', 0.0):.6f}")
    
    # Gradient Norms (Unclipped / Clipped)
    gu = optim_stats.get('gu', -1.0)
    gc = optim_stats.get('gc', 0.0)
    grad_str = f"GU:{gu:.2f} GC:{gc:.2f}" if gu != -1 else f"G:{gc:.2f}"
    status_table.add_row("Gradients", grad_str)

    # Task Stats (Time T and Length L)
    if 't_val' in task_stats:
        t_str = f"µ{task_stats['t_val']:.1f}±{task_stats['t_std']:.1f} [{task_stats['t_min']}-{task_stats['t_max']}]"
        status_table.add_row("Time (T)", t_str)
    
    if 'l_mean' in task_stats:
        l_str = f"µ{task_stats['l_mean']:.1f}±{task_stats['l_std']:.1f} [{task_stats['l_min']}-{task_stats['l_max']}]"
        status_table.add_row("Len (L)", l_str)
    
    # Gradient Pulse Panel (Feeder Check)
    grad_panel = Panel(
        f"Core:      [green]{grad_stats.get('core', 0.0):.5f}[/]\n"
        f"Periphery: [yellow]{grad_stats.get('periphery', 0.0):.5f}[/]",
        title="Topology Grads",
        border_style="blue"
    )

    # Vital Signs Panel
    vitals_content = (
        f"Energy: [bold]{vitals_stats.get('energy', 0.0):.4f}[/]\n"
        f"Dead:   [red]{vitals_stats.get('dead', 0.0):.1f}%[/]\n"
        f"Rank:   [cyan]{vitals_stats.get('rank', 0.0):.1f}[/]\n"
        f"Wiggle: [bold blue]{vitals_stats.get('wiggle_index', 0.0):.2f}[/]\n"
        f"Lateral:   [bold yellow]{vitals_stats.get('ring_strength', 0.0):.4f}[/]\n"
        f"Leak:   [magenta]{vitals_stats.get('fast_core', 0.0):.1f}%[/]\n"
        f"[dim]──────────────[/]\n"
        f"r(Core): [bold green]{vitals_stats.get('r_core_mean', 0.0):.2f}[/]±[dim]{vitals_stats.get('r_core_std', 0.0):.2f}[/]\n"
        f"r(Periphery): [bold cyan]{vitals_stats.get('r_peri_mean', 0.0):.2f}[/]±[dim]{vitals_stats.get('r_peri_std', 0.0):.2f}[/]"
    )
    vitals_panel = Panel(vitals_content, title="System Vitals", border_style="red")

    # Layout
    layout = Layout()
    layout.split_column(
        Layout(name="upper", ratio=1),
        Layout(pbar)
    )
    layout["upper"].split_row(
        Layout(Panel(status_table, title="Training Stats"), ratio=1),
        Layout(grad_panel, ratio=1),
        Layout(vitals_panel, ratio=1)
    )

    return layout


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Selection
    parser.add_argument('--model', type=str, required=True, choices=['ctm', 'lstm', 'ff'], help='Model type to train.')

    # Model Architecture
    # Common across all or most
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--backbone_type', type=str, default='resnet34-2', help='Type of backbone featureiser.') # Default changed from original script
    # CTM / LSTM specific
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input (CTM, LSTM).')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads (CTM, LSTM).') # Default changed
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks (CTM, LSTM).')
    parser.add_argument('--positional_embedding_type', type=str, default='none',
                        help='Type of positional embedding (CTM, LSTM).', choices=['none',
                                                                       'learnable-fourier',
                                                                       'multi-learnable-fourier',
                                                                       'custom-rotational'])

    # CTM specific
    parser.add_argument('--synapse_depth', type=int, default=8, help='Depth of U-NET model for synapse. 1=linear, no unet (CTM only).') # Default changed
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons to use for output synch (CTM only).') # Default changed
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons to use for observation/action synch (CTM only).') # Default changed
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing', help='Protocol for selecting neuron subset (CTM only).')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Number of neurons paired self-to-self for synch (CTM only).')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of the pre-activation history for NLMS (CTM only).')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Use deep memory (CTM only).')
    parser.add_argument('--memory_hidden_dims', type=int, default=32, help='Hidden dimensions of the memory if using deep memory (CTM only).') # Default changed
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model (CTM only).')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs (CTM only).')
    # LSTM specific
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM stacked layers (LSTM only).') # Added LSTM arg

    # Task Specific Args (Common to all models for this task)
    parser.add_argument('--maze_route_length', type=int, default=100, help='Length to truncate targets.')
    parser.add_argument('--cirriculum_lookahead', type=int, default=5, help='How far to look ahead for cirriculum.')


    # Training
    parser.add_argument('--expand_range', action=argparse.BooleanOptionalAction, default=True, help='Mazes between 0 and 1 = False. Between -1 and 1 = True. Legacy checkpoints use 0 and 1.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.') # Default changed
    parser.add_argument('--batch_size_test', type=int, default=64, help='Batch size for testing.') # Default changed
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the model.') # Default changed
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[], help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--num_workers_train', type=int, default=0, help='Num workers training.') # Renamed from num_workers, kept default
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False, help='Try to compile model components.')

    # Logging and Saving
    parser.add_argument('--log_dir', type=str, default='logs/scratch', help='Directory for logging.')
    parser.add_argument('--dataset', type=str, default='mazes-medium', help='Dataset to use.', choices=['mazes-medium', 'mazes-large', 'mazes-small']) 
    parser.add_argument('--data_root', type=str, default='data/mazes', help='Data root.')
    
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False, help='Reload only the model from disk?')
    parser.add_argument('--strict_reload', action=argparse.BooleanOptionalAction, default=True, help='Should use strict reload for model weights.') # Added back
    parser.add_argument('--ignore_metrics_when_reloading', action=argparse.BooleanOptionalAction, default=False, help='Ignore metrics when reloading (for debugging)?') # Added back

    # Tracking
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=20, help='How many minibatches to approx metrics. Set to -1 for full eval') # Default changed

    # Device
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='List of GPU(s) to use. Set to -1 to use CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')


    args = parser.parse_args()
    return args


if __name__=='__main__':

    # Hosuekeeping
    args = parse_args()

    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    diag_dir = f"{args.log_dir}/sw_diagnostics"
    if not os.path.exists(diag_dir):
        os.makedirs(diag_dir)

    assert args.dataset in ['mazes-medium', 'mazes-large', 'mazes-small']

    

    prediction_reshaper = [args.maze_route_length, 5]  # Problem specific 
    args.out_dims = args.maze_route_length * 5 # Output dimension before reshaping

    # For total reproducibility
    zip_python_code(f'{args.log_dir}/repo_state.zip')
    with open(f'{args.log_dir}/args.txt', 'w') as f:
        print(args, file=f)

    # Configure device string (support MPS on macOS)
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model {args.model} on {device}')


    # Build model conditionally
    model = None
    if args.model == 'ctm':
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper, 
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
        ).to(device)
    elif args.model == 'lstm':
         model = LSTMBaseline(
            num_layers=args.num_layers,
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads, 
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper, 
            dropout=args.dropout,
        ).to(device)
    elif args.model == 'ff':
        model = FFBaseline(
            d_model=args.d_model,
            backbone_type=args.backbone_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    try:
        # Determine pseudo input shape based on dataset
        h_w = 39 if args.dataset in ['mazes-small', 'mazes-medium'] else 99 # Example dimensions
        pseudo_inputs = torch.zeros((1, 3, h_w, h_w), device=device).float()
        model(pseudo_inputs)
    except Exception as e:
         print(f"Warning: Pseudo forward pass failed: {e}")

    print(f'Total params: {sum(p.numel() for p in model.parameters())}')

    # Data
    dataset_mean = [0,0,0]  # For plotting later
    dataset_std = [1,1,1]

    which_maze = args.dataset.split('-')[-1]
    data_root = f'{args.data_root}/{which_maze}'

    train_data = MazeImageFolder(root=f'{data_root}/train/', which_set='train', maze_route_length=args.maze_route_length, expand_range=args.expand_range)
    test_data = MazeImageFolder(root=f'{data_root}/test/', which_set='test', maze_route_length=args.maze_route_length, expand_range=args.expand_range)

    num_workers_test = 1 # Defaulting to 1, can be changed
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_train, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=num_workers_test, drop_last=False)
    train_eval_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.batch_size_test,
        shuffle=True,
        num_workers=num_workers_test,
        drop_last=True,
        persistent_workers=True
    )
    test_eval_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.batch_size_test,
        shuffle=True, 
        num_workers=num_workers_test,
        drop_last=True,
        persistent_workers=True
    )

    # For lazy modules so that we can get param count
    

    model.train()

    # Optimizer and scheduler
    decay_params = []
    no_decay_params = []
    special_lr_params = [] # Group 3: High LR, No Weight Decay (Small-World)
    
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check for Small-World Decay Params FIRST (High LR)
        if "decay_params" in name:
            special_lr_params.append(param)
            continue
        if any(exclusion_str in name for exclusion_str in args.weight_decay_exclusion_list):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
    if len(no_decay_names):
        print(f'WARNING, excluding: {no_decay_names}')

    # Construct Parameter Groups
    param_groups = []
    
    if len(decay_params) > 0:
        param_groups.append({'params': decay_params, 'weight_decay': args.weight_decay})
        
    if len(no_decay_params) > 0:
        param_groups.append({'params': no_decay_params, 'weight_decay': 0.0})
        
    if len(special_lr_params) > 0:
        param_groups.append({'params': special_lr_params, 'weight_decay': 0.0, 'lr': args.lr * 1.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr, # Default LR for groups that don't specify it
        eps=1e-8 if not args.use_amp else 1e-6
    )

    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
    if args.use_scheduler:
        if args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)
        else:
            raise NotImplementedError


    # Metrics tracking
    start_iter = 0
    train_losses = []
    test_losses = []
    train_accuracies = []  # Per tick/step accuracy list
    test_accuracies = []   
    train_accuracies_most_certain = [] # Accuracy, fine-grained
    test_accuracies_most_certain = []  
    train_accuracies_most_certain_permaze = [] # Full maze accuracy
    test_accuracies_most_certain_permaze = []  
    iters = []

    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)
    if args.reload:
        checkpoint_path = f'{args.log_dir}/checkpoint.pt'
        if os.path.isfile(checkpoint_path):
            print(f'Reloading from: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not args.strict_reload: print('WARNING: not using strict reload for model weights!')
            load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=args.strict_reload)
            print(f" Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

            if not args.reload_model_only:
                print('Reloading optimizer etc.')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict']) # Load scaler state
                start_iter = checkpoint['iteration']

                if not args.ignore_metrics_when_reloading:
                    train_losses = checkpoint['train_losses']
                    test_losses = checkpoint['test_losses']
                    train_accuracies = checkpoint['train_accuracies']
                    test_accuracies = checkpoint['test_accuracies']
                    iters = checkpoint['iters']
                    train_accuracies_most_certain = checkpoint['train_accuracies_most_certain']
                    test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
                    train_accuracies_most_certain_permaze = checkpoint['train_accuracies_most_certain_permaze']
                    test_accuracies_most_certain_permaze = checkpoint['test_accuracies_most_certain_permaze']
                else:
                     print("Ignoring metrics history upon reload.")

            else:
                print('Only reloading model!')

            if 'torch_rng_state' in checkpoint:
                # Reset seeds
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu().byte())
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])

            del checkpoint
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.do_compile:
        print('Compiling...')
        if hasattr(model, 'backbone'):
            model.backbone = torch.compile(model.backbone, mode='default', fullgraph=True)
        # Compile synapses and NLMs only for CTM
        if args.model == 'ctm':
            model.synapses = torch.compile(model.synapses, mode='default', fullgraph=True)
            model.trace_processor = torch.compile(model.trace_processor, mode='default', fullgraph=True)

    if args.model == 'ctm' and hasattr(model, 'out_neuron_indices_left'):
        export_full_network(model, save_path=f"{args.log_dir}/sw_full_network.graphml")
        visualize_topology_matrix(model, save_path=f"{args.log_dir}/topology_matrix_global.png")

    # Training
    grad_stats = {'core': 0.0, 'periphery': 0.0}
    vitals_stats = {
        'r_core_mean': 0.0, 'r_core_std': 0.0,
        'r_peri_mean': 0.0, 'r_peri_std': 0.0,
        'energy': 0.0, 'dead': 0.0,
        'rank': 0.0, 'fast_core': 0.0,
        'ring_strength': 0.0, 'wiggle_index': 0.0,
    }
    optim_stats = {'lr': 0.0, 'gu': -1.0, 'gc': 0.0}
    task_stats = {} # Will be populated dynamically
    
    current_loss = 0.0
    current_acc = 0.0
    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task_id = job_progress.add_task("[green]Training...[/]", total=args.training_iterations)
    iterator = iter(trainloader)
    with Live(refresh_per_second=4) as live:
        for bi in range(start_iter, args.training_iterations):
            optim_stats['lr'] = optimizer.param_groups[0]['lr']

            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            inputs = inputs.to(device)
            targets = targets.to(device) # Shape (B, SeqLength)

            # All for nice metric printing:
            loss = None
            accuracy_finegrained = None # Per-step accuracy at chosen tick
            where_most_certain_val = -1.0 # Default value
            where_most_certain_std = 0.0
            where_most_certain_min = -1
            where_most_certain_max = -1
            upto_where_mean = -1.0
            upto_where_std = 0.0
            upto_where_min = -1
            upto_where_max = -1


            # Model-specific forward, reshape, and loss calculation
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                # if args.do_compile: # CUDAGraph marking applied if compiling any model
                #      torch.compiler.cudagraph_mark_step_begin()

                if args.model == 'ctm':
                    # CTM output: (B, SeqLength*5, Ticks), Certainties: (B, Ticks)
                    predictions_raw, certainties, synchronisation = model(inputs)
                    # Reshape predictions: (B, SeqLength, 5, Ticks)
                    predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1))
                    loss, where_most_certain, upto_where = maze_loss(predictions, certainties, targets, cirriculum_lookahead=args.cirriculum_lookahead, use_most_certain=True)
                    # Accuracy uses predictions[B, S, C, T] indexed at where_most_certain[B] -> gives (B, S, C) -> argmax(2) -> (B,S)
                    accuracy_finegrained = (predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] == targets).float().mean().item()

                elif args.model == 'lstm':
                    # LSTM output: (B, SeqLength*5, Ticks), Certainties: (B, Ticks)
                    predictions_raw, certainties, synchronisation = model(inputs)
                     # Reshape predictions: (B, SeqLength, 5, Ticks)
                    predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1))
                    loss, where_most_certain, upto_where = maze_loss(predictions, certainties, targets, cirriculum_lookahead=args.cirriculum_lookahead, use_most_certain=False)
                    # where_most_certain should be -1 (last tick) here. Accuracy calc follows same logic.
                    accuracy_finegrained = (predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] == targets).float().mean().item()

                elif args.model == 'ff':
                    # Assume FF output: (B, SeqLength*5)
                    predictions_raw = model(inputs)
                    # Reshape predictions: (B, SeqLength, 5)
                    predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5)
                    # FF has no certainties, pass None. maze_loss must handle this.
                    # Unsqueeze predictions for compatibility with maze loss calcluation
                    loss, where_most_certain, upto_where = maze_loss(predictions.unsqueeze(-1), None, targets, cirriculum_lookahead=args.cirriculum_lookahead, use_most_certain=False)
                    # where_most_certain should be -1 here. Accuracy uses 3D prediction tensor.
                    accuracy_finegrained = (predictions.argmax(2) == targets).float().mean().item()


                # Extract stats from loss outputs if they are tensors
                if torch.is_tensor(where_most_certain):
                    task_stats['t_val'] = where_most_certain.float().mean().item()
                    task_stats['t_std'] = where_most_certain.float().std().item()
                    task_stats['t_min'] = where_most_certain.min().item()
                    task_stats['t_max'] = where_most_certain.max().item()

                if isinstance(upto_where, (np.ndarray, list)) and len(upto_where) > 0:
                    task_stats['l_mean'] = np.mean(upto_where)
                    task_stats['l_std'] = np.std(upto_where)
                    task_stats['l_min'] = np.min(upto_where)
                    task_stats['l_max'] = np.max(upto_where)


            scaler.scale(loss).backward()

            # --- Gradient Pulse  ---
            # Assessing if Feeder Lines (p=3.0) are receiving credit or vanishing.
            if bi % 100 == 0:
                with torch.no_grad():
                    if hasattr(model, 'decay_params_out') and model.decay_params_out.grad is not None:
                        d_params = model.decay_params_out.grad
                        left = model.out_neuron_indices_left
                        right = model.out_neuron_indices_right
                        
                        self_loop_mask = (left == right)
                        hub_indices = torch.unique(left[self_loop_mask])
                        
                        # Core = Source is Hub AND Target is Hub
                        is_source_hub = torch.isin(left, hub_indices)
                        is_target_hub = torch.isin(right, hub_indices)
                        is_core = is_source_hub & is_target_hub

                        scale = scaler.get_scale()
                        grad_stats['core'] = d_params[is_core].norm().item() / scale
                        grad_stats['periphery'] = d_params[~is_core].norm().item() / scale

            optim_stats['gu'] = -1.0
            if args.gradient_clipping != -1:
                scaler.unscale_(optimizer)
                unclipped_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        unclipped_norm += p.grad.detach().data.norm(2).item() ** 2
                optim_stats['gu'] = unclipped_norm ** 0.5
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)

            scaler.step(optimizer)
            
            # --- Monitor Clipped Gradient ---
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.detach().data.norm(2).item() ** 2
            optim_stats['gc'] = grad_norm ** 0.5

            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            with torch.no_grad():
                # Handle torch.compile or DDP wrapping
                real_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                real_model = real_model.module if hasattr(real_model, 'module') else real_model

                if hasattr(real_model, 'decay_params_out'):
                    left = real_model.out_neuron_indices_left
                    right = real_model.out_neuron_indices_right
                    
                    self_loop_mask = (left == right)
                    hub_indices = torch.unique(left[self_loop_mask])
                    
                    is_source_hub = torch.isin(left, hub_indices)
                    is_target_hub = torch.isin(right, hub_indices)
                    
                    # CORE = Source is Hub AND Target is Hub
                    # This captures both Self-Loops (H->H) and Ring (H->Next H)
                    mask_core = is_source_hub & is_target_hub
                    
                    # PERIPHERY = Source is NOT a Hub (Input -> Hub)
                    mask_peri = ~mask_core
                    
                    # Core (Memory + Ring): Force Slow Decay (0.001 - 0.4)
                    real_model.decay_params_out[mask_core] = real_model.decay_params_out[mask_core].clamp(min=0.001, max=0.4)
                    
                    # Periphery (Feeders): Force Fast Decay (> 0.6)
                    real_model.decay_params_out[mask_peri] = real_model.decay_params_out[mask_peri].clamp(min=0.6, max=3.0)

                # Prevent explosion for action parameters
                if hasattr(real_model, 'decay_params_action'):
                    real_model.decay_params_action.clamp_(min=0.001)

            # Metrics tracking and plotting
            if bi%args.track_every==0 and (bi != 0 or args.reload_model_only):
                job_progress.update(task_id, description="[yellow]Evaluating...[/]")
                model.eval() # Use eval mode for consistency during tracking
                with torch.inference_mode(): # Use inference mode for tracking

                    # --- Quantitative Metrics ---
                    iters.append(bi)
                    # Re-initialize metric lists for this evaluation step
                    current_train_losses_eval = []
                    current_test_losses_eval = []
                    current_train_accuracies_eval = []
                    current_test_accuracies_eval = []
                    current_train_accuracies_most_certain_eval = []
                    current_test_accuracies_most_certain_eval = []
                    current_train_accuracies_most_certain_permaze_eval = []
                    current_test_accuracies_most_certain_permaze_eval = []

                    # TRAIN METRICS
                    all_targets_list = []
                    all_predictions_list = [] # Per step/tick predictions argmax (N, S, T) or (N, S)
                    all_predictions_most_certain_list = [] # Predictions at chosen step/tick argmax (N, S)
                    all_losses = []

                    if 1:  # To avoid de-indenting due to removed tqdm ctx
                        for inferi, (inputs, targets) in enumerate(train_eval_loader):
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            all_targets_list.append(targets.detach().cpu().numpy()) # N x S

                            # Model-specific forward, reshape, loss for evaluation
                            if args.model == 'ctm':
                                predictions_raw, certainties, _ = model(inputs)
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                                loss, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=True)
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S,C,T -> argmax class -> B,S,T
                                pred_at_certain = predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] # B,S
                                all_predictions_most_certain_list.append(pred_at_certain.detach().cpu().numpy())

                            elif args.model == 'lstm':
                                predictions_raw, certainties, _ = model(inputs)
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                                loss, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=False) # where = -1
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S,C,T
                                pred_at_certain = predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] # B,S (at last tick)
                                all_predictions_most_certain_list.append(pred_at_certain.detach().cpu().numpy())

                            elif args.model == 'ff':
                                predictions_raw = model(inputs) # B, S*C
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5) # B,S,C
                                loss, where_most_certain, _ = maze_loss(predictions.unsqueeze(-1), None, targets, use_most_certain=False) # where = -1
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S
                                all_predictions_most_certain_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S (same as above for FF)


                            all_losses.append(loss.item())

                            if args.n_test_batches != -1 and inferi >= args.n_test_batches -1 : break

                    all_targets = np.concatenate(all_targets_list) # N, S
                    all_predictions = np.concatenate(all_predictions_list) # N, S, T or N, S
                    all_predictions_most_certain = np.concatenate(all_predictions_most_certain_list) # N, S

                    train_losses.append(np.mean(all_losses))
                    # Calculate per step/tick accuracy averaged over batches
                    if args.model in ['ctm', 'lstm']:
                         # all_predictions shape (N, S, T), all_targets shape (N, S) -> compare targets to each tick prediction
                         train_accuracies.append(np.mean(all_predictions == all_targets[:,:,np.newaxis], axis=0)) # Mean over N -> (S, T)
                    else: # FF
                         # all_predictions shape (N, S), all_targets shape (N, S)
                         train_accuracies.append(np.mean(all_predictions == all_targets, axis=0)) # Mean over N -> (S,)

                    # Calculate accuracy at chosen step/tick ("most certain") averaged over all steps and batches
                    train_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean()) # Scalar
                    # Calculate full maze accuracy at chosen step/tick averaged over batches
                    train_accuracies_most_certain_permaze.append((all_targets == all_predictions_most_certain).reshape(all_targets.shape[0], -1).all(-1).mean()) # Scalar


                    # TEST METRICS
                    all_targets_list = []
                    all_predictions_list = []
                    all_predictions_most_certain_list = []
                    all_losses = []

                    if 1:  # To avoid de-indenting due to removed tqdm ctx
                        for inferi, (inputs, targets) in enumerate(test_eval_loader):
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            all_targets_list.append(targets.detach().cpu().numpy())

                             # Model-specific forward, reshape, loss for evaluation
                            if args.model == 'ctm':
                                predictions_raw, certainties, _ = model(inputs)
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                                loss, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=True)
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S,T
                                pred_at_certain = predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] # B,S
                                all_predictions_most_certain_list.append(pred_at_certain.detach().cpu().numpy())

                            elif args.model == 'lstm':
                                predictions_raw, certainties, _ = model(inputs)
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                                loss, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=False) # where = -1
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S,T
                                pred_at_certain = predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] # B,S (at last tick)
                                all_predictions_most_certain_list.append(pred_at_certain.detach().cpu().numpy())

                            elif args.model == 'ff':
                                predictions_raw = model(inputs) # B, S*C
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5) # B,S,C
                                loss, where_most_certain, _ = maze_loss(predictions.unsqueeze(-1), None, targets, use_most_certain=False) # where = -1
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S
                                all_predictions_most_certain_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S (same as above for FF)


                            all_losses.append(loss.item())

                            if args.n_test_batches != -1 and inferi >= args.n_test_batches -1: break

                    all_targets = np.concatenate(all_targets_list)
                    all_predictions = np.concatenate(all_predictions_list)
                    all_predictions_most_certain = np.concatenate(all_predictions_most_certain_list)

                    test_losses.append(np.mean(all_losses))
                    # Calculate per step/tick accuracy
                    if args.model in ['ctm', 'lstm']:
                         test_accuracies.append(np.mean(all_predictions == all_targets[:,:,np.newaxis], axis=0)) # -> (S, T)
                    else: # FF
                         test_accuracies.append(np.mean(all_predictions == all_targets, axis=0)) # -> (S,)

                    # Calculate "most certain" accuracy
                    test_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean()) # Scalar
                    # Calculate full maze accuracy
                    test_accuracies_most_certain_permaze.append((all_targets == all_predictions_most_certain).reshape(all_targets.shape[0], -1).all(-1).mean()) # Scalar


                    # --- Plotting ---
                    # Accuracy Plot (Handling different dimensions)
                    figacc = plt.figure(figsize=(10, 10))
                    axacc_train = figacc.add_subplot(211)
                    axacc_test = figacc.add_subplot(212)
                    cm = sns.color_palette("viridis", as_cmap=True)

                    # Plot per step/tick accuracy
                    # train_accuracies is List[(S, T)] or List[(S,)]
                    # We need to average over S dimension for plotting
                    train_acc_plot = [np.mean(acc_s) for acc_s in train_accuracies] # List[Scalar] or List[Scalar] after mean
                    test_acc_plot = [np.mean(acc_s) for acc_s in test_accuracies]   # List[Scalar] or List[Scalar] after mean

                    axacc_train.plot(iters, train_acc_plot, 'g-', alpha=0.5, label='Avg Step Acc')
                    axacc_test.plot(iters, test_acc_plot, 'g-', alpha=0.5, label='Avg Step Acc')


                    # Plot most certain accuracy 
                    axacc_train.plot(iters, train_accuracies_most_certain, 'k--', alpha=0.7, label='Most Certain (Avg Step)')
                    axacc_test.plot(iters, test_accuracies_most_certain, 'k--', alpha=0.7, label='Most Certain (Avg Step)')
                    # Plot full maze accuracy 
                    axacc_train.plot(iters, train_accuracies_most_certain_permaze, 'r-', alpha=0.6, label='Full Maze')
                    axacc_test.plot(iters, test_accuracies_most_certain_permaze, 'r-', alpha=0.6, label='Full Maze')

                    axacc_train.set_title('Train Accuracy')
                    axacc_test.set_title('Test Accuracy')
                    axacc_train.legend(loc='lower right')
                    axacc_test.legend(loc='lower right')
                    axacc_train.set_xlim([0, args.training_iterations])
                    axacc_test.set_xlim([0, args.training_iterations])
                    axacc_train.set_ylim([0, 1]) # Set Ylim for accuracy
                    axacc_test.set_ylim([0, 1])

                    figacc.tight_layout()
                    figacc.savefig(f'{args.log_dir}/accuracies.png', dpi=150)
                    plt.close(figacc)

                    # Loss Plot
                    figloss = plt.figure(figsize=(10, 5))
                    axloss = figloss.add_subplot(111)
                    axloss.plot(iters, train_losses, 'b-', linewidth=1, alpha=0.8, label=f'Train: {train_losses[-1]:.4f}')
                    axloss.plot(iters, test_losses, 'r-', linewidth=1, alpha=0.8, label=f'Test: {test_losses[-1]:.4f}')
                    axloss.legend(loc='upper right')
                    axloss.set_xlim([0, args.training_iterations])
                    axloss.set_ylim(bottom=0) 

                    figloss.tight_layout()
                    figloss.savefig(f'{args.log_dir}/losses.png', dpi=150)
                    plt.close(figloss)

                    # --- Visualization Section (Conditional) ---
                    if args.model in ['ctm', 'lstm']:
                        #  try:
                            inputs_viz, targets_viz = next(iter(testloader))
                            inputs_viz = inputs_viz.to(device)
                            targets_viz = targets_viz.to(device)
                            # Find longest path in batch for potentially better visualization
                            longest_index = (targets_viz!=4).sum(-1).argmax() # Action 4 assumed padding/end

                            # Track internal states
                            predictions_viz_raw, certainties_viz, (synch_out_viz, _), pre_activations_viz, post_activations_viz, attention_tracking_viz = model(inputs_viz, track=True)

                            # Reshape predictions (assuming raw is B, D, T)
                            predictions_viz = predictions_viz_raw.reshape(predictions_viz_raw.size(0), -1, 5, predictions_viz_raw.size(-1)) # B, S, C, T

                            att_shape = (model.kv_features.shape[2], model.kv_features.shape[3])
                            attention_tracking_viz = attention_tracking_viz.reshape(
                                attention_tracking_viz.shape[0], 
                                attention_tracking_viz.shape[1], -1, att_shape[0], att_shape[1])

                            # Plot dynamics (common plotting function)
                            plot_neural_dynamics(post_activations_viz, 100, args.log_dir, axis_snap=True)

                            # Create maze GIF (task-specific plotting)
                            wiggle_index = make_maze_gif((inputs_viz[longest_index].detach().cpu().numpy()+1)/2,
                                          predictions_viz[longest_index].detach().cpu().numpy(), # Pass reshaped B,S,C,T -> S,C,T
                                          targets_viz[longest_index].detach().cpu().numpy(), # S
                                          attention_tracking_viz[:, longest_index],  # Pass T, (H), H, W
                                          args.log_dir)

                            # --- Small-World Health Check ---
                            if hasattr(model, 'out_neuron_indices_left'):
                                decay = np.exp(-model.decay_params_out.detach().cpu().numpy())
                                left = model.out_neuron_indices_left.detach().cpu().numpy()
                                right = model.out_neuron_indices_right.detach().cpu().numpy()
                                activity_np = synch_out_viz  # (Time, Batch, Edges)

                                # Identify Core Memory (Self-Loops: i -> i)
                                mask_self_loops = (left == right)
                                
                                # Identify Hub Indices from Self-Loops
                                if mask_self_loops.any():
                                    hub_indices = np.unique(left[mask_self_loops])
                                else:
                                    hub_indices = np.array([])

                                # Identify Ring Connections (Hub -> Hub, but not i -> i)
                                is_source_hub = np.isin(left, hub_indices)
                                is_target_hub = np.isin(right, hub_indices)
                                mask_ring = is_source_hub & is_target_hub & (~mask_self_loops)

                                # Identify Feeders (Non-Hub -> Hub)
                                # Strictly speaking, anything that originates from a non-hub is a feeder
                                mask_feeders = (~is_source_hub)

                                # View A: For Rank & Dead Check (Batch, Edges)
                                # We average over Time (Axis 0) to get the "Average Thought" per sample
                                activity_over_time = np.abs(activity_np).mean(axis=0) 
                                
                                # View B: For Ring Integrity & Plots (Time, Edges)
                                # We average over Batch (Axis 1) to get the "Trajectory"
                                activity_over_batch = np.abs(activity_np).mean(axis=1)

                                # --- Calculate Vitals ---
                                
                                # A. Ring Integrity
                                # Measures if Hubs are actually talking to each other
                                if mask_ring.any():
                                    ring_integrity = np.abs(activity_over_batch[:, mask_ring]).mean()
                                else:
                                    ring_integrity = 0.0

                                # B. Energy/Death
                                mean_act = activity_over_time.mean()
                                dead_pct = (activity_over_time.mean(axis=0) < 1e-6).mean() * 100
                                
                                # C. Effective Rank (The Blob Check)
                                if mask_self_loops.any():
                                    target_activity = activity_over_time[:, mask_self_loops] # (Batch, Hubs)
                                else:
                                    target_activity = activity_over_time
                                try:
                                    _, S, _ = np.linalg.svd(target_activity)
                                    S_norm = S / (S.sum() + 1e-9)
                                    effective_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-9)))
                                    vitals_stats['singular_values'] = S_norm
                                except:
                                    effective_rank = 0.0
                                    vitals_stats['singular_values'] = np.zeros(10)

                                # D. Granular Decay Rates (Latency Risk Check)
                                # We check drift on Self-Loops (Memory Stability)
                                if mask_self_loops.any():
                                    r_core = decay[mask_self_loops]
                                    r_core_mean = r_core.mean().item()
                                    r_core_std = r_core.std().item()
                                    fast_core_pct = (np.sum(r_core < 0.67) / len(r_core)) * 100
                                else:
                                    r_core_mean, r_core_std, fast_core_pct = 0.0, 0.0, 0.0

                                if mask_feeders.any():
                                    r_peri = decay[mask_feeders]
                                    r_peri_mean = r_peri.mean().item()
                                    r_peri_std = r_peri.std().item()
                                else:
                                    r_peri_mean, r_peri_std = 0.0, 0.0

                                # Update State for Dashboard
                                vitals_stats.update({
                                    'r_core_mean': r_core_mean, 'r_core_std': r_core_std,
                                    'r_peri_mean': r_peri_mean, 'r_peri_std': r_peri_std,
                                    'fast_core': fast_core_pct,
                                    'energy': mean_act,
                                    'dead': dead_pct,
                                    'rank': effective_rank,
                                    'ring_strength': ring_integrity,
                                    'wiggle_index': wiggle_index,
                                })

                                visualize_ctm_dashboard(
                                    model,
                                    activity_over_batch,
                                    vitals_stats, 
                                    f"{args.log_dir}/sw_diagnostics/sw", 
                                    bi
                                )
                        #  except Exception as e:
                        #       print(f"Visualization failed for model {args.model}: {e}")
                    # --- End Visualization ---

                job_progress.update(task_id, description="[green]Training...[/]")
                model.train() # Switch back to train mode
            
            # --- UI UPDATE ---
            current_loss = loss.item()
            current_acc = accuracy_finegrained
            job_progress.update(task_id, advance=1)
            
            # Re-render the dashboard
            live.update(make_dashboard(
                bi, args.training_iterations, 
                current_loss, current_acc, 
                grad_stats, 
                vitals_stats, 
                optim_stats, 
                task_stats, 
                job_progress
            ))


            # Save model checkpoint
            if (bi % args.save_every == 0 or bi == args.training_iterations - 1) and bi != start_iter:
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(), # Save scaler state
                    'iteration': bi,
                    # Save all tracked metrics
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracies': train_accuracies, # List of (S, T) or (S,) arrays
                    'test_accuracies': test_accuracies,   # List of (S, T) or (S,) arrays
                    'train_accuracies_most_certain': train_accuracies_most_certain, # List of scalars
                    'test_accuracies_most_certain': test_accuracies_most_certain,   # List of scalars
                    'train_accuracies_most_certain_permaze': train_accuracies_most_certain_permaze, # List of scalars
                    'test_accuracies_most_certain_permaze': test_accuracies_most_certain_permaze,   # List of scalars
                    'iters': iters,
                    'args': args, # Save args used for this run
                    # RNG states
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                }
                torch.save(checkpoint_data, f'{args.log_dir}/checkpoint.pt')
