#!/usr/bin/env python
"""
Gradient Conflict Analysis Script for VAD Multi-Task Learning

This script analyzes gradient conflicts between different tasks (planning, detection, map, motion)
in the shared BEV encoder parameters. It computes pairwise cosine similarity between task gradients
to diagnose potential gradient conflicts before applying PCGrad or similar methods.

VAD architecture: shared BEV encoder (BEVFormerEncoder) + separate task-specific decoders.
Unlike HiP-AD's unified OneDecoder, the shared parameters here are in the BEV encoder
and optionally the backbone (ResNet50 + FPN).

Analysis modes:
  basic     - Overall pairwise cosine similarity across all shared parameters
  full      - Per-layer analysis + advanced metrics (interference, decomposition)
  direction - PCA-based gradient direction visualization + all of 'full'

Usage:
    # Single checkpoint (64 samples = 1 weight update with 8 GPUs × 8 batch)
    python tools/analyze_gradient_conflict.py \\
        --checkpoint data/ckpts/epoch_60.pth \\
        --num-batches 64 --batch-size 1 \\
        --analysis-mode full \\
        --output-dir gradient_analysis_results

    # Multi-checkpoint epoch dynamics
    python tools/analyze_gradient_conflict.py \\
        --checkpoints data/ckpts/epoch_1.pth data/ckpts/epoch_10.pth \\
                      data/ckpts/epoch_20.pth data/ckpts/epoch_30.pth \\
                      data/ckpts/epoch_40.pth data/ckpts/epoch_50.pth \\
                      data/ckpts/epoch_60.pth \\
        --checkpoint-labels ep1 ep10 ep20 ep30 ep40 ep50 ep60 \\
        --num-batches 64 --batch-size 1 \\
        --output-dir gradient_analysis_results
"""

import os
import csv
import sys
import json
import math
import pickle
import tempfile
import argparse
import contextlib
from functools import partial
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats as sp_stats

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, MMDataParallel
from mmdet.models import build_detector

# Import VAD custom modules (triggers registration)
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset


# ============================================================================
# Constants
# ============================================================================

TASK_GROUPS = {
    'det': ['loss_cls', 'loss_bbox'],
    'map': ['loss_map_cls', 'loss_map_bbox', 'loss_map_iou', 'loss_map_pts', 'loss_map_dir'],
    'motion': ['loss_traj', 'loss_traj_cls'],
    'plan': ['loss_plan_reg', 'loss_plan_bound', 'loss_plan_col', 'loss_plan_dir'],
}

TASK_PAIRS = [
    ('plan', 'det'),
    ('plan', 'map'),
    ('plan', 'motion'),
    ('det', 'map'),
    ('det', 'motion'),
    ('map', 'motion'),
]

TASK_COLORS = {
    'det': '#e74c3c',
    'map': '#2ecc71',
    'motion': '#3498db',
    'plan': '#f39c12',
}


# ============================================================================
# Utility Functions
# ============================================================================

@contextlib.contextmanager
def _selective_eval(model):
    """Forward dispatch는 train 유지, Dropout/BN만 eval로 전환."""
    switched = []
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                          nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.training:
                m.eval()
                switched.append(m)
    try:
        yield
    finally:
        for m in switched:
            m.train()


def match_loss_key(loss_key: str, task_prefixes: List[str]) -> bool:
    """Match loss key against task prefixes, handling dN. intermediate prefixes.

    E.g., 'd0.loss_cls' matches 'loss_cls' (det), but 'd0.loss_map_cls' does NOT
    match 'loss_cls' — it matches 'loss_map_cls' (map).
    """
    # Strip dN. prefix for intermediate decoder layer losses
    stripped = loss_key
    if loss_key.startswith('d') and '.' in loss_key:
        prefix_part = loss_key.split('.', 1)[0]
        if prefix_part[1:].isdigit():
            stripped = loss_key.split('.', 1)[1]

    # Strip enc_ prefix for encoder losses
    if stripped.startswith('enc_'):
        stripped = stripped[4:]

    for prefix in task_prefixes:
        if stripped == prefix or stripped.startswith(prefix):
            return True
    return False


def compute_cosine_similarity(grad1: torch.Tensor, grad2: torch.Tensor) -> float:
    if grad1 is None or grad2 is None:
        return float('nan')
    norm1 = torch.norm(grad1)
    norm2 = torch.norm(grad2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return float('nan')
    return (torch.dot(grad1, grad2) / (norm1 * norm2)).item()


def compute_interference(grad_source: torch.Tensor, grad_victim: torch.Tensor) -> float:
    """Interference of source on victim: ||g_src|| * max(0, -cos(g_src, g_vic))."""
    cos = compute_cosine_similarity(grad_source, grad_victim)
    if np.isnan(cos):
        return float('nan')
    return torch.norm(grad_source).item() * max(0.0, -cos)


def compute_decomposition(grad_a: torch.Tensor, grad_b: torch.Tensor) -> Tuple[float, float]:
    """Decompose grad_a into cooperative and conflicting components w.r.t. grad_b.

    Returns (cooperative_magnitude, conflicting_magnitude).
    """
    if grad_a is None or grad_b is None:
        return (float('nan'), float('nan'))
    norm_b = torch.norm(grad_b)
    if norm_b < 1e-8:
        return (float('nan'), float('nan'))
    proj_scalar = torch.dot(grad_a, grad_b) / (norm_b * norm_b)
    proj_mag = proj_scalar.item() * norm_b.item()
    if proj_mag >= 0:
        return (proj_mag, 0.0)
    else:
        return (0.0, abs(proj_mag))


def move_to_device(data: dict, device: str) -> dict:
    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, list):
            result[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        elif isinstance(value, dict):
            result[key] = move_to_device(value, device)
        else:
            result[key] = value
    return result


# ============================================================================
# Model / BEV Encoder Helpers
# ============================================================================

def get_bev_encoder(model: nn.Module):
    """Get the shared BEV encoder from VAD model."""
    if hasattr(model, 'module'):
        model = model.module
    return model.pts_bbox_head.transformer.encoder


def get_backbone_modules(model: nn.Module) -> dict:
    """Get backbone (ResNet) and neck (FPN) modules."""
    if hasattr(model, 'module'):
        model = model.module
    return {
        'img_backbone': model.img_backbone,
        'img_neck': model.img_neck,
    }


# ============================================================================
# Grouped Parameter Extraction
# ============================================================================

def get_shared_parameters_grouped(
    model: nn.Module,
    shared_layer_names: List[str],
    analysis_target: str = 'bev_encoder',
) -> Tuple[OrderedDict, OrderedDict]:
    """Extract shared parameters grouped by (encoder_layer, operation_type).

    Args:
        model: The VAD model (possibly wrapped in MMDataParallel).
        shared_layer_names: Filter for operation types, e.g.
            ['temporal_self_attn', 'spatial_cross_attn', 'ffn', 'norm']
        analysis_target: 'bev_encoder', 'backbone', or 'all'

    Returns:
        param_groups: OrderedDict[group_key -> OrderedDict[param_name -> Parameter]]
        group_meta:   OrderedDict[group_key -> dict] with keys:
                        encoder_layer_idx, op_type
    """
    param_groups = OrderedDict()
    group_meta = OrderedDict()

    # --- BEV Encoder parameters ---
    if analysis_target in ('bev_encoder', 'all'):
        encoder = get_bev_encoder(model)

        # Map operation types to sub-module accessors within BEVFormerLayer
        # BEVFormerLayer has: attentions (ModuleList), ffns (ModuleList), norms (ModuleList)
        # operation_order: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        # attentions[0] = TemporalSelfAttention (self_attn)
        # attentions[1] = SpatialCrossAttention (cross_attn)
        # ffns[0] = FFN
        # norms[0,1,2] = LayerNorm

        op_mapping = {
            'temporal_self_attn': ('attentions', 0),
            'spatial_cross_attn': ('attentions', 1),
            'ffn': ('ffns', 0),
        }

        for layer_idx, layer in enumerate(encoder.layers):
            # Attention and FFN modules
            for op_name, (attr_name, sub_idx) in op_mapping.items():
                if op_name not in shared_layer_names:
                    continue
                module_list = getattr(layer, attr_name, None)
                if module_list is None or sub_idx >= len(module_list):
                    continue
                sub_module = module_list[sub_idx]

                group_key = f"bev{layer_idx}_{op_name}"
                params = OrderedDict()
                for name, param in sub_module.named_parameters():
                    if param.requires_grad:
                        params[f"encoder.layers.{layer_idx}.{attr_name}.{sub_idx}.{name}"] = param

                if params:
                    param_groups[group_key] = params
                    group_meta[group_key] = {
                        'encoder_layer_idx': layer_idx,
                        'op_type': op_name,
                    }

            # Norm layers
            if 'norm' in shared_layer_names:
                norms = getattr(layer, 'norms', None)
                if norms is not None:
                    for norm_idx, norm_module in enumerate(norms):
                        group_key = f"bev{layer_idx}_norm_{norm_idx}"
                        params = OrderedDict()
                        for name, param in norm_module.named_parameters():
                            if param.requires_grad:
                                params[f"encoder.layers.{layer_idx}.norms.{norm_idx}.{name}"] = param
                        if params:
                            param_groups[group_key] = params
                            group_meta[group_key] = {
                                'encoder_layer_idx': layer_idx,
                                'op_type': f'norm_{norm_idx}',
                            }

    # --- Backbone parameters ---
    if analysis_target in ('backbone', 'all'):
        raw_model = model.module if hasattr(model, 'module') else model
        backbone = raw_model.img_backbone
        neck = raw_model.img_neck

        # ResNet layers (frozen_stages=1 means layer0,1 are frozen)
        for stage_name in ['layer2', 'layer3', 'layer4']:
            stage = getattr(backbone, stage_name, None)
            if stage is None:
                continue
            group_key = f"backbone_{stage_name}"
            params = OrderedDict()
            for name, param in stage.named_parameters():
                if param.requires_grad:
                    params[f"img_backbone.{stage_name}.{name}"] = param
            if params:
                param_groups[group_key] = params
                group_meta[group_key] = {
                    'encoder_layer_idx': -1,
                    'op_type': stage_name,
                }

        # FPN
        group_key = "neck_fpn"
        params = OrderedDict()
        for name, param in neck.named_parameters():
            if param.requires_grad:
                params[f"img_neck.{name}"] = param
        if params:
            param_groups[group_key] = params
            group_meta[group_key] = {
                'encoder_layer_idx': -1,
                'op_type': 'fpn',
            }

    return param_groups, group_meta


def get_shared_parameters(model, shared_layer_names, analysis_target='bev_encoder'):
    """Backward-compatible flat parameter dict."""
    param_groups, _ = get_shared_parameters_grouped(model, shared_layer_names, analysis_target)
    flat = OrderedDict()
    for gk, params in param_groups.items():
        flat.update(params)
    return flat


# ============================================================================
# Gradient Computation
# ============================================================================

def _sum_task_loss(loss_dict, task_name):
    """Sum all losses for a given task."""
    prefixes = TASK_GROUPS.get(task_name, [])
    task_loss = None
    for key, val in loss_dict.items():
        if match_loss_key(key, prefixes) and isinstance(val, torch.Tensor) and val.requires_grad:
            task_loss = val if task_loss is None else task_loss + val
    return task_loss


def compute_task_gradient_grouped(
    model: nn.Module,
    loss_dict: Dict[str, torch.Tensor],
    task_name: str,
    param_groups: OrderedDict,
    retain_graph: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    """Compute per-group gradient vectors for a task. Single autograd.grad call.

    Returns dict mapping group_key -> flattened gradient tensor (on same device).
    Also includes '_all' key for full concatenated gradient.
    Returns None if no matching losses.
    """
    task_loss = _sum_task_loss(loss_dict, task_name)
    if task_loss is None:
        return None

    # Build ordered list of all parameters across groups
    all_params = []
    for gk, params in param_groups.items():
        all_params.extend(params.values())

    if not all_params:
        return None

    try:
        grads = torch.autograd.grad(
            outputs=task_loss,
            inputs=all_params,
            retain_graph=retain_graph,
            allow_unused=True,
            create_graph=False,
        )
    except RuntimeError as e:
        print(f"Warning: gradient computation failed for {task_name}: {e}")
        return None

    # Replace None grads with zeros
    device = all_params[0].device
    dtype = all_params[0].dtype
    grad_flat = []
    for idx, g in enumerate(grads):
        if g is not None:
            grad_flat.append(g.detach().flatten())
        else:
            grad_flat.append(torch.zeros(all_params[idx].numel(), device=device, dtype=dtype))

    full_grad = torch.cat(grad_flat)

    # Split into groups
    result = {}
    cum = 0
    for gk, params in param_groups.items():
        n = sum(p.numel() for p in params.values())
        result[gk] = full_grad[cum:cum + n]
        cum += n
    result['_all'] = full_grad

    return result


# ============================================================================
# Per-Group Metrics
# ============================================================================

def compute_per_group_metrics(task_grads_grouped, group_keys):
    """Compute cosine, norms, interference, decomposition for all groups and task pairs."""
    tasks = list(task_grads_grouped.keys())
    per_group_cosine = {}
    per_group_norms = {}
    per_group_conflict = {}
    per_group_interference = {}
    per_group_decomposition = {}

    for gk in group_keys:
        cosines = {}
        norms = {}
        conflicts = {}
        interferences = {}
        decompositions = {}

        for task in tasks:
            g = task_grads_grouped[task].get(gk)
            norms[task] = torch.norm(g).item() if g is not None else 0.0

        for t1, t2 in TASK_PAIRS:
            pk = f"{t1}_vs_{t2}"
            g1 = task_grads_grouped.get(t1, {}).get(gk)
            g2 = task_grads_grouped.get(t2, {}).get(gk)
            if g1 is not None and g2 is not None:
                c = compute_cosine_similarity(g1, g2)
                cosines[pk] = c
                conflicts[pk] = c < 0 if not np.isnan(c) else False
                interferences[f"{t1}_on_{t2}"] = compute_interference(g1, g2)
                interferences[f"{t2}_on_{t1}"] = compute_interference(g2, g1)
                coop, conf = compute_decomposition(g1, g2)
                decompositions[f"{t1}_wrt_{t2}"] = (coop, conf)
                coop2, conf2 = compute_decomposition(g2, g1)
                decompositions[f"{t2}_wrt_{t1}"] = (coop2, conf2)

        per_group_cosine[gk] = cosines
        per_group_norms[gk] = norms
        per_group_conflict[gk] = conflicts
        per_group_interference[gk] = interferences
        per_group_decomposition[gk] = decompositions

    return per_group_cosine, per_group_norms, per_group_conflict, per_group_interference, per_group_decomposition


# ============================================================================
# Single Batch Analysis
# ============================================================================

def analyze_single_batch(
    model, data, param_groups, group_meta, device,
    fp16=False, analysis_mode='full', store_gradients=False,
    selective_eval=True,
):
    results = {
        'pairwise_cosine': {},
        'gradient_norms': {},
        'conflict_flags': {},
    }

    model.train()

    _ctx = _selective_eval(model) if selective_eval else contextlib.nullcontext()
    with _ctx:
        with torch.cuda.amp.autocast(enabled=fp16):
            # VAD forward: model(**data) handles temporal splitting internally
            # MMDataParallel handles DataContainer scatter
            outputs = model(**data)

        if not isinstance(outputs, dict):
            print("Warning: Model output is not a dictionary of losses")
            return results

        task_names = list(TASK_GROUPS.keys())
        task_grads_grouped = {}
        group_keys = [gk for gk in param_groups.keys()]

        for i, task in enumerate(task_names):
            retain = (i < len(task_names) - 1)
            grad_dict = compute_task_gradient_grouped(model, outputs, task, param_groups, retain_graph=retain)
            if grad_dict is not None:
                # Move to CPU immediately to save GPU memory
                task_grads_grouped[task] = {gk: g.cpu() for gk, g in grad_dict.items()}
                results['gradient_norms'][task] = torch.norm(grad_dict['_all']).item()

    # Overall pairwise cosine
    for t1, t2 in TASK_PAIRS:
        pk = f"{t1}_vs_{t2}"
        g1 = task_grads_grouped.get(t1, {}).get('_all')
        g2 = task_grads_grouped.get(t2, {}).get('_all')
        if g1 is not None and g2 is not None:
            c = compute_cosine_similarity(g1, g2)
            results['pairwise_cosine'][pk] = c
            results['conflict_flags'][pk] = c < 0 if not np.isnan(c) else False

    # Overall interference & decomposition
    if analysis_mode in ('full', 'direction'):
        interference = {}
        decomposition = {}
        for t1, t2 in TASK_PAIRS:
            g1 = task_grads_grouped.get(t1, {}).get('_all')
            g2 = task_grads_grouped.get(t2, {}).get('_all')
            if g1 is not None and g2 is not None:
                interference[f"{t1}_on_{t2}"] = compute_interference(g1, g2)
                interference[f"{t2}_on_{t1}"] = compute_interference(g2, g1)
                c1, f1 = compute_decomposition(g1, g2)
                decomposition[f"{t1}_wrt_{t2}"] = (c1, f1)
                c2, f2 = compute_decomposition(g2, g1)
                decomposition[f"{t2}_wrt_{t1}"] = (c2, f2)
        results['interference'] = interference
        results['decomposition'] = decomposition

    # Per-group metrics
    if analysis_mode in ('full', 'direction'):
        pgc, pgn, pgf, pgi, pgd = compute_per_group_metrics(task_grads_grouped, group_keys)
        results['per_group_cosine'] = pgc
        results['per_group_norms'] = pgn
        results['per_group_conflict'] = pgf
        results['per_group_interference'] = pgi
        results['per_group_decomposition'] = pgd

    # Store raw gradients for PCA
    if store_gradients:
        results['task_gradients_all'] = {t: task_grads_grouped[t]['_all'] for t in task_grads_grouped}
        results['task_gradients_grouped'] = {
            t: {gk: g for gk, g in task_grads_grouped[t].items() if gk != '_all'}
            for t in task_grads_grouped
        }

    model.zero_grad(set_to_none=True)
    return results


# ============================================================================
# Aggregation
# ============================================================================

def _agg_scalar_dict(all_results, key):
    """Aggregate a key that maps to {sub_key: float} across batches."""
    agg = defaultdict(list)
    for r in all_results:
        d = r.get(key, {})
        for sk, val in d.items():
            if isinstance(val, (int, float)) and not np.isnan(val):
                agg[sk].append(val)
    return dict(agg)


def _stats_from_values(values):
    if not values:
        return None
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'values': values,
    }


def aggregate_results(all_results, analysis_mode='basic'):
    stats = {
        'pairwise_cosine': {},
        'gradient_norms': {},
        'conflict_frequency': {},
        'num_batches': len(all_results),
    }

    cos_agg = defaultdict(list)
    norm_agg = defaultdict(list)
    flag_agg = defaultdict(list)
    for r in all_results:
        for pk, c in r.get('pairwise_cosine', {}).items():
            if not np.isnan(c):
                cos_agg[pk].append(c)
        for t, n in r.get('gradient_norms', {}).items():
            norm_agg[t].append(n)
        for pk, f in r.get('conflict_flags', {}).items():
            flag_agg[pk].append(f)

    for pk, vals in cos_agg.items():
        stats['pairwise_cosine'][pk] = _stats_from_values(vals)
    for t, vals in norm_agg.items():
        stats['gradient_norms'][t] = _stats_from_values(vals)
    for pk, vals in flag_agg.items():
        ratio = sum(vals) / len(vals)
        stats['conflict_frequency'][pk] = {'ratio': ratio, 'count': sum(vals), 'total': len(vals)}

    if analysis_mode in ('full', 'direction'):
        # Interference
        intf_agg = _agg_scalar_dict(all_results, 'interference')
        stats['interference'] = {k: _stats_from_values(v) for k, v in intf_agg.items() if v}

        # Decomposition
        decomp_coop = defaultdict(list)
        decomp_conf = defaultdict(list)
        for r in all_results:
            for dk, (c, f) in r.get('decomposition', {}).items():
                if not np.isnan(c):
                    decomp_coop[dk].append(c)
                    decomp_conf[dk].append(f)
        stats['decomposition'] = {}
        for dk in decomp_coop:
            stats['decomposition'][dk] = {
                'cooperative': _stats_from_values(decomp_coop[dk]),
                'conflicting': _stats_from_values(decomp_conf[dk]),
            }

        # Per-group
        pg_cos_agg = defaultdict(lambda: defaultdict(list))
        pg_norm_agg = defaultdict(lambda: defaultdict(list))
        pg_flag_agg = defaultdict(lambda: defaultdict(list))
        pg_intf_agg = defaultdict(lambda: defaultdict(list))
        pg_coop_agg = defaultdict(lambda: defaultdict(list))
        pg_conf_agg = defaultdict(lambda: defaultdict(list))

        for r in all_results:
            for gk, cosines in r.get('per_group_cosine', {}).items():
                for pk, c in cosines.items():
                    if not np.isnan(c):
                        pg_cos_agg[gk][pk].append(c)
            for gk, norms in r.get('per_group_norms', {}).items():
                for t, n in norms.items():
                    pg_norm_agg[gk][t].append(n)
            for gk, flags in r.get('per_group_conflict', {}).items():
                for pk, f in flags.items():
                    pg_flag_agg[gk][pk].append(f)
            for gk, intfs in r.get('per_group_interference', {}).items():
                for ik, v in intfs.items():
                    if not np.isnan(v):
                        pg_intf_agg[gk][ik].append(v)
            for gk, decomps in r.get('per_group_decomposition', {}).items():
                for dk, (c, f) in decomps.items():
                    if not np.isnan(c):
                        pg_coop_agg[gk][dk].append(c)
                        pg_conf_agg[gk][dk].append(f)

        stats['per_group_cosine'] = {
            gk: {pk: _stats_from_values(vals) for pk, vals in pks.items()}
            for gk, pks in pg_cos_agg.items()
        }
        stats['per_group_norms'] = {
            gk: {t: _stats_from_values(vals) for t, vals in ts.items()}
            for gk, ts in pg_norm_agg.items()
        }
        stats['per_group_conflict_frequency'] = {}
        for gk, pks in pg_flag_agg.items():
            stats['per_group_conflict_frequency'][gk] = {}
            for pk, vals in pks.items():
                ratio = sum(vals) / len(vals)
                stats['per_group_conflict_frequency'][gk][pk] = {'ratio': ratio, 'count': sum(vals), 'total': len(vals)}
        stats['per_group_interference'] = {
            gk: {ik: _stats_from_values(vals) for ik, vals in iks.items()}
            for gk, iks in pg_intf_agg.items()
        }
        stats['per_group_decomposition'] = {}
        for gk in pg_coop_agg:
            stats['per_group_decomposition'][gk] = {}
            for dk in pg_coop_agg[gk]:
                stats['per_group_decomposition'][gk][dk] = {
                    'cooperative': _stats_from_values(pg_coop_agg[gk][dk]),
                    'conflicting': _stats_from_values(pg_conf_agg[gk][dk]),
                }

    # Gradient directions (for PCA)
    if analysis_mode == 'direction':
        grad_dirs = defaultdict(list)
        for r in all_results:
            for t, g in r.get('task_gradients_all', {}).items():
                grad_dirs[t].append(g)
        stats['gradient_directions'] = dict(grad_dirs)

        grad_dirs_grouped = defaultdict(lambda: defaultdict(list))
        for r in all_results:
            for t, groups in r.get('task_gradients_grouped', {}).items():
                for gk, g in groups.items():
                    grad_dirs_grouped[t][gk].append(g)
        stats['gradient_directions_grouped'] = {
            t: dict(groups) for t, groups in grad_dirs_grouped.items()
        }

    return stats


# ============================================================================
# Save & Print
# ============================================================================

def save_results(stats, output_dir, analysis_mode='basic'):
    os.makedirs(output_dir, exist_ok=True)

    def strip_values(d):
        if isinstance(d, dict):
            return {k: strip_values(v) for k, v in d.items() if k != 'values'}
        return d

    with open(os.path.join(output_dir, 'gradient_conflict_stats.json'), 'w') as f:
        json.dump(strip_values(stats), f, indent=2, default=str)

    with open(os.path.join(output_dir, 'gradient_conflict_details.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task Pair', 'Mean Cosine', 'Std Cosine', 'Min', 'Max', 'Conflict Ratio'])
        for pk, cs in stats.get('pairwise_cosine', {}).items():
            cf = stats.get('conflict_frequency', {}).get(pk, {})
            w.writerow([pk, f"{cs['mean']:.4f}", f"{cs['std']:.4f}",
                        f"{cs['min']:.4f}", f"{cs['max']:.4f}", f"{cf.get('ratio', 0):.4f}"])

    with open(os.path.join(output_dir, 'gradient_norms.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task', 'Mean Norm', 'Std Norm', 'Min', 'Max'])
        for t, ns in stats.get('gradient_norms', {}).items():
            w.writerow([t, f"{ns['mean']:.4f}", f"{ns['std']:.4f}", f"{ns['min']:.4f}", f"{ns['max']:.4f}"])

    if analysis_mode in ('full', 'direction') and 'per_group_cosine' in stats:
        layer_dir = os.path.join(output_dir, 'per_layer')
        os.makedirs(layer_dir, exist_ok=True)
        with open(os.path.join(layer_dir, 'per_layer_details.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Group', 'Task Pair', 'Mean Cosine', 'Std Cosine', 'Conflict Ratio'])
            for gk, pks in sorted(stats['per_group_cosine'].items()):
                for pk, cs in sorted(pks.items()):
                    cf = stats.get('per_group_conflict_frequency', {}).get(gk, {}).get(pk, {})
                    w.writerow([gk, pk, f"{cs['mean']:.4f}", f"{cs['std']:.4f}", f"{cf.get('ratio', 0):.4f}"])

    print(f"Results saved to {output_dir}")


def print_summary(stats, analysis_mode='basic', focus_task='plan'):
    print("\n" + "=" * 70)
    print("GRADIENT CONFLICT ANALYSIS SUMMARY (VAD)")
    print("=" * 70)
    print(f"\nAnalyzed {stats['num_batches']} batches")

    print("\n--- Pairwise Cosine Similarity (Mean +/- Std) ---")
    for pk, cs in sorted(stats.get('pairwise_cosine', {}).items()):
        cf = stats.get('conflict_frequency', {}).get(pk, {})
        ratio = cf.get('ratio', 0)
        marker = " << CONFLICT" if ratio > 0.3 else ""
        print(f"  {pk:20s}: {cs['mean']:+.4f} +/- {cs['std']:.4f}  "
              f"(conflict rate: {ratio * 100:.1f}%){marker}")

    print("\n--- Gradient Norms (Mean +/- Std) ---")
    for t, ns in sorted(stats.get('gradient_norms', {}).items()):
        print(f"  {t:10s}: {ns['mean']:.4f} +/- {ns['std']:.4f}")

    norms = stats.get('gradient_norms', {})
    if focus_task in norms:
        focus_norm = norms[focus_task]['mean']
        for t, ns in norms.items():
            if t != focus_task and focus_norm > 0:
                ratio = ns['mean'] / focus_norm
                if ratio > 5:
                    print(f"  WARNING: {t} norm is {ratio:.1f}x larger than {focus_task}")

    if analysis_mode in ('full', 'direction'):
        intf = stats.get('interference', {})
        if intf:
            print(f"\n--- Interference on '{focus_task}' (higher = worse) ---")
            for t in ['det', 'map', 'motion']:
                key = f"{t}_on_{focus_task}"
                if key in intf:
                    print(f"  {t:10s} -> {focus_task}: {intf[key]['mean']:.4f} +/- {intf[key]['std']:.4f}")

        pg_cos = stats.get('per_group_cosine', {})
        if pg_cos:
            print(f"\n--- Top 10 Conflicting (group, pair) ---")
            all_entries = []
            for gk, pks in pg_cos.items():
                for pk, cs in pks.items():
                    all_entries.append((gk, pk, cs['mean']))
            all_entries.sort(key=lambda x: x[2])
            for gk, pk, mean_cos in all_entries[:10]:
                cf = stats.get('per_group_conflict_frequency', {}).get(gk, {}).get(pk, {})
                ratio = cf.get('ratio', 0)
                print(f"  {gk:30s} | {pk:20s} | cos={mean_cos:+.4f} | conflict={ratio * 100:.1f}%")

    print("\n--- Conflict Assessment ---")
    high = [(k, v['ratio']) for k, v in stats.get('conflict_frequency', {}).items() if v['ratio'] > 0.3]
    if high:
        print("  High conflict pairs (>30% negative cosine):")
        for pair, ratio in sorted(high, key=lambda x: -x[1]):
            print(f"    - {pair}: {ratio * 100:.1f}%")
    else:
        print("  No significant gradient conflicts detected.")

    print("=" * 70 + "\n")


# ============================================================================
# Visualizations — Basic
# ============================================================================

def create_basic_visualizations(stats, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Skipping visualizations.")
        return

    os.makedirs(output_dir, exist_ok=True)
    task_names = ['plan', 'det', 'map', 'motion']

    # 1. Heatmap
    n = len(task_names)
    mat = np.ones((n, n))
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            if i != j:
                pk = f"{t1}_vs_{t2}" if f"{t1}_vs_{t2}" in stats['pairwise_cosine'] else f"{t2}_vs_{t1}"
                if pk in stats['pairwise_cosine']:
                    mat[i, j] = stats['pairwise_cosine'][pk]['mean']
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=task_names, yticklabels=task_names, vmin=-1, vmax=1)
    plt.title('VAD: Mean Cosine Similarity between Task Gradients\n(Negative = Conflict)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_heatmap.png'), dpi=150)
    plt.close()

    # 2. Conflict frequency bar chart
    pairs = list(stats.get('conflict_frequency', {}).keys())
    if pairs:
        ratios = [stats['conflict_frequency'][p]['ratio'] for p in pairs]
        colors = ['#e74c3c' if r > 0.5 else '#f39c12' if r > 0.2 else '#2ecc71' for r in ratios]
        plt.figure(figsize=(10, 5))
        plt.bar(pairs, ratios, color=colors)
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
        plt.xlabel('Task Pair')
        plt.ylabel('Conflict Frequency (cosine < 0)')
        plt.title('VAD: Gradient Conflict Frequency by Task Pair')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'conflict_frequency.png'), dpi=150)
        plt.close()

    # 3. Box plot
    data_plot, labels = [], []
    for pk in stats.get('pairwise_cosine', {}):
        vals = stats['pairwise_cosine'][pk].get('values', [])
        if vals:
            data_plot.append(vals)
            labels.append(pk)
    if data_plot:
        plt.figure(figsize=(12, 5))
        plt.boxplot(data_plot, labels=labels, patch_artist=True)
        plt.axhline(y=0, color='r', linestyle='--', label='Conflict threshold')
        plt.xlabel('Task Pair')
        plt.ylabel('Cosine Similarity')
        plt.title('VAD: Distribution of Gradient Cosine Similarities')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cosine_distribution.png'), dpi=150)
        plt.close()

    # 4. Gradient norms
    tasks = list(stats.get('gradient_norms', {}).keys())
    if tasks:
        means = [stats['gradient_norms'][t]['mean'] for t in tasks]
        stds = [stats['gradient_norms'][t]['std'] for t in tasks]
        plt.figure(figsize=(8, 5))
        plt.bar(tasks, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
        plt.xlabel('Task')
        plt.ylabel('Gradient Norm')
        plt.title('VAD: Mean Gradient Norm by Task')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_norms.png'), dpi=150)
        plt.close()

    print(f"Basic visualizations saved to {output_dir}")


# ============================================================================
# Visualizations — Per-Layer
# ============================================================================

def create_per_layer_visualizations(stats, group_meta, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed.")
        return

    layer_dir = os.path.join(output_dir, 'per_layer')
    os.makedirs(layer_dir, exist_ok=True)

    pg_cos = stats.get('per_group_cosine', {})
    pg_norms = stats.get('per_group_norms', {})
    pg_cf = stats.get('per_group_conflict_frequency', {})
    if not pg_cos:
        return

    sorted_gks = sorted(pg_cos.keys(), key=lambda gk: (
        group_meta.get(gk, {}).get('encoder_layer_idx', 99),
        group_meta.get(gk, {}).get('op_type', ''),
    ))
    pair_keys = [f"{t1}_vs_{t2}" for t1, t2 in TASK_PAIRS]

    # --- Per-Operation Conflict Heatmap ---
    mat = np.full((len(sorted_gks), len(pair_keys)), np.nan)
    for i, gk in enumerate(sorted_gks):
        for j, pk in enumerate(pair_keys):
            cs = pg_cos.get(gk, {}).get(pk)
            if cs:
                mat[i, j] = cs['mean']

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_gks) * 0.35)))
    sns.heatmap(mat, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                xticklabels=pair_keys, yticklabels=sorted_gks,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title('VAD: Per-Operation Cosine Similarity (Negative = Conflict)')
    plt.tight_layout()
    plt.savefig(os.path.join(layer_dir, 'per_operation_conflict_heatmap.png'), dpi=150)
    plt.close()

    # --- Per-BEV-Layer Conflict (aggregated) ---
    enc_indices = sorted(set(
        group_meta[gk]['encoder_layer_idx'] for gk in sorted_gks
        if group_meta.get(gk, {}).get('encoder_layer_idx', -1) >= 0
    ))
    has_backbone = any(group_meta.get(gk, {}).get('encoder_layer_idx', -1) < 0 for gk in sorted_gks)

    layer_labels = [f"bev{d}" for d in enc_indices]
    if has_backbone:
        layer_labels.append("backbone")

    mat_layer = np.full((len(layer_labels), len(pair_keys)), np.nan)
    mat_cf = np.full((len(layer_labels), len(pair_keys)), np.nan)

    for di, enc_idx in enumerate(enc_indices):
        enc_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('encoder_layer_idx') == enc_idx]
        for pj, pk in enumerate(pair_keys):
            vals = []
            cf_vals = []
            for gk in enc_gks:
                cs = pg_cos.get(gk, {}).get(pk)
                if cs:
                    vals.extend(cs.get('values', [cs['mean']]))
                cf = pg_cf.get(gk, {}).get(pk)
                if cf:
                    cf_vals.append(cf['ratio'])
            if vals:
                mat_layer[di, pj] = float(np.mean(vals))
            if cf_vals:
                mat_cf[di, pj] = float(np.mean(cf_vals))

    if has_backbone:
        bb_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('encoder_layer_idx', -1) < 0]
        fi = len(layer_labels) - 1
        for pj, pk in enumerate(pair_keys):
            vals = []
            cf_vals = []
            for gk in bb_gks:
                cs = pg_cos.get(gk, {}).get(pk)
                if cs:
                    vals.extend(cs.get('values', [cs['mean']]))
                cf = pg_cf.get(gk, {}).get(pk)
                if cf:
                    cf_vals.append(cf['ratio'])
            if vals:
                mat_layer[fi, pj] = float(np.mean(vals))
            if cf_vals:
                mat_cf[fi, pj] = float(np.mean(cf_vals))

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(layer_labels) * 0.5)))
    sns.heatmap(mat_layer, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                xticklabels=pair_keys, yticklabels=layer_labels,
                vmin=-1, vmax=1, ax=axes[0], linewidths=0.5)
    axes[0].set_title('Per-BEV-Layer Mean Cosine')

    sns.heatmap(mat_cf, annot=True, fmt='.1%', cmap='Reds',
                xticklabels=pair_keys, yticklabels=layer_labels,
                vmin=0, vmax=1, ax=axes[1], linewidths=0.5)
    axes[1].set_title('Per-BEV-Layer Conflict Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(layer_dir, 'per_bev_layer_conflict.png'), dpi=150)
    plt.close()

    # --- Operation Type Conflict Profile ---
    op_types = sorted(set(group_meta[gk]['op_type'] for gk in sorted_gks))
    mat_op = np.full((len(op_types), len(pair_keys)), np.nan)
    for oi, op in enumerate(op_types):
        op_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('op_type') == op]
        for pj, pk in enumerate(pair_keys):
            vals = []
            for gk in op_gks:
                cs = pg_cos.get(gk, {}).get(pk)
                if cs:
                    vals.extend(cs.get('values', [cs['mean']]))
            if vals:
                mat_op[oi, pj] = float(np.mean(vals))

    plt.figure(figsize=(10, max(4, len(op_types) * 0.6)))
    sns.heatmap(mat_op, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=pair_keys, yticklabels=op_types,
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title('VAD: Operation Type Conflict Profile')
    plt.tight_layout()
    plt.savefig(os.path.join(layer_dir, 'operation_type_conflict.png'), dpi=150)
    plt.close()

    # --- Per-Layer Gradient Norm Dominance ---
    if pg_norms:
        task_names = ['det', 'map', 'motion', 'plan']
        available_tasks = set()
        for gk in sorted_gks:
            available_tasks.update(pg_norms.get(gk, {}).keys())
        task_names = [t for t in task_names if t in available_tasks]

        x = np.arange(len(sorted_gks))
        width = 0.8 / len(task_names)

        fig, ax = plt.subplots(figsize=(max(12, len(sorted_gks) * 0.6), 6))
        for ti, task in enumerate(task_names):
            means = []
            for gk in sorted_gks:
                ns = pg_norms.get(gk, {}).get(task)
                means.append(ns['mean'] if ns else 0)
            ax.bar(x + ti * width, means, width, label=task, color=TASK_COLORS.get(task, 'gray'), alpha=0.8)

        ax.set_xticks(x + width * len(task_names) / 2)
        ax.set_xticklabels(sorted_gks, rotation=90, ha='center', fontsize=7)
        ax.set_ylabel('Gradient Norm')
        ax.set_title('VAD: Per-Layer Gradient Norm Dominance')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, 'gradient_norm_dominance.png'), dpi=150)
        plt.close()

    # --- Gradient Norm Ratio Heatmap ---
    if pg_norms:
        mat_ratio = np.full((len(sorted_gks), len(pair_keys)), np.nan)
        for i, gk in enumerate(sorted_gks):
            for j, pk in enumerate(pair_keys):
                t1, t2 = pk.split('_vs_')
                n1 = pg_norms.get(gk, {}).get(t1)
                n2 = pg_norms.get(gk, {}).get(t2)
                if n1 and n2 and n1['mean'] > 1e-10 and n2['mean'] > 1e-10:
                    mat_ratio[i, j] = np.log2(n1['mean'] / n2['mean'])

        plt.figure(figsize=(10, max(6, len(sorted_gks) * 0.35)))
        sns.heatmap(mat_ratio, annot=True, fmt='.1f', cmap='coolwarm', center=0,
                    xticklabels=pair_keys, yticklabels=sorted_gks, linewidths=0.5)
        plt.title('VAD: Gradient Norm Ratio log2(task1/task2) per Layer')
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, 'gradient_norm_ratio.png'), dpi=150)
        plt.close()

    print(f"Per-layer visualizations saved to {layer_dir}")


# ============================================================================
# Visualizations — Advanced
# ============================================================================

def create_advanced_visualizations(stats, group_meta, output_dir, focus_task='plan'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed.")
        return

    adv_dir = os.path.join(output_dir, 'advanced')
    os.makedirs(adv_dir, exist_ok=True)

    task_names = ['det', 'map', 'motion', 'plan']

    # --- Interference Magnitude (asymmetric heatmap) ---
    intf = stats.get('interference', {})
    if intf:
        mat = np.zeros((len(task_names), len(task_names)))
        for i, src in enumerate(task_names):
            for j, vic in enumerate(task_names):
                if i != j:
                    key = f"{src}_on_{vic}"
                    if key in intf:
                        mat[i, j] = intf[key]['mean']

        plt.figure(figsize=(8, 6))
        sns.heatmap(mat, annot=True, fmt='.3f', cmap='Reds',
                    xticklabels=[f"{t}\n(victim)" for t in task_names],
                    yticklabels=[f"{t}\n(source)" for t in task_names],
                    linewidths=0.5)
        plt.title('VAD: Gradient Interference Magnitude\n||g_src|| * max(0, -cos(g_src, g_vic))')
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'interference_magnitude.png'), dpi=150)
        plt.close()

    # --- Cooperative vs Conflicting Decomposition per BEV layer ---
    pg_decomp = stats.get('per_group_decomposition', {})
    pg_cos = stats.get('per_group_cosine', {})
    if pg_decomp:
        sorted_gks = sorted(pg_cos.keys(), key=lambda gk: (
            group_meta.get(gk, {}).get('encoder_layer_idx', 99),
            group_meta.get(gk, {}).get('op_type', ''),
        ))

        enc_indices = sorted(set(
            group_meta[gk]['encoder_layer_idx'] for gk in sorted_gks
            if group_meta.get(gk, {}).get('encoder_layer_idx', -1) >= 0
        ))

        aux_tasks = [t for t in task_names if t != focus_task]
        fig, axes = plt.subplots(1, len(aux_tasks), figsize=(6 * len(aux_tasks), 5))
        if len(aux_tasks) == 1:
            axes = [axes]

        for ai, aux in enumerate(aux_tasks):
            dk = f"{aux}_wrt_{focus_task}"
            coop_vals = []
            conf_vals = []
            dec_labels = []
            for d in enc_indices:
                dec_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('encoder_layer_idx') == d]
                coop_sum, conf_sum, cnt = 0, 0, 0
                for gk in dec_gks:
                    decomp = pg_decomp.get(gk, {}).get(dk)
                    if decomp:
                        coop_sum += decomp['cooperative']['mean']
                        conf_sum += decomp['conflicting']['mean']
                        cnt += 1
                if cnt > 0:
                    coop_vals.append(coop_sum / cnt)
                    conf_vals.append(conf_sum / cnt)
                    dec_labels.append(f"bev{d}")

            x = np.arange(len(dec_labels))
            axes[ai].bar(x, coop_vals, 0.6, label='Cooperative', color='#2ecc71', alpha=0.8)
            axes[ai].bar(x, [-c for c in conf_vals], 0.6, label='Conflicting', color='#e74c3c', alpha=0.8)
            axes[ai].set_xticks(x)
            axes[ai].set_xticklabels(dec_labels)
            axes[ai].axhline(0, color='k', linewidth=0.5)
            axes[ai].set_title(f'{aux} gradient w.r.t. {focus_task}')
            axes[ai].set_ylabel('Component Magnitude')
            axes[ai].legend()

        plt.suptitle(f'VAD: Cooperative vs Conflicting Gradient Components (focus: {focus_task})', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'cooperative_vs_conflicting.png'), dpi=150)
        plt.close()

    # --- Cumulative Gradient Flow ---
    if pg_cos:
        sorted_gks = sorted(pg_cos.keys(), key=lambda gk: (
            group_meta.get(gk, {}).get('encoder_layer_idx', 99),
            group_meta.get(gk, {}).get('op_type', ''),
        ))
        enc_indices = sorted(set(
            group_meta[gk]['encoder_layer_idx'] for gk in sorted_gks
            if group_meta.get(gk, {}).get('encoder_layer_idx', -1) >= 0
        ))

        fig, ax = plt.subplots(figsize=(10, 5))
        for pk in [f"{t1}_vs_{t2}" for t1, t2 in TASK_PAIRS]:
            vals = []
            for d in enc_indices:
                dec_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('encoder_layer_idx') == d]
                batch_vals = []
                for gk in dec_gks:
                    cs = pg_cos.get(gk, {}).get(pk)
                    if cs:
                        batch_vals.extend(cs.get('values', [cs['mean']]))
                vals.append(float(np.mean(batch_vals)) if batch_vals else np.nan)

            ax.plot(enc_indices, vals, 'o-', label=pk, linewidth=2, markersize=6)

        ax.axhline(0, color='r', linestyle='--', linewidth=0.8, label='Conflict threshold')
        ax.set_xlabel('BEV Encoder Layer')
        ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title('VAD: Gradient Conflict Across BEV Encoder Depth')
        ax.legend(fontsize=8, ncol=2)
        ax.set_xticks(enc_indices)
        ax.set_xticklabels([f"bev{d}" for d in enc_indices])
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'cumulative_gradient_flow.png'), dpi=150)
        plt.close()

    # --- Planning-Centric Dashboard ---
    if pg_cos and intf:
        aux_tasks = [t for t in task_names if t != focus_task]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        enc_indices_valid = sorted(set(
            group_meta[gk]['encoder_layer_idx'] for gk in pg_cos.keys()
            if group_meta.get(gk, {}).get('encoder_layer_idx', -1) >= 0
        ))

        ax = axes[0, 0]
        for aux in aux_tasks:
            pk = f"{focus_task}_vs_{aux}" if f"{focus_task}_vs_{aux}" in list(pg_cos.values())[0] else f"{aux}_vs_{focus_task}"
            vals = []
            for d in enc_indices_valid:
                dec_gks = [gk for gk in pg_cos if group_meta.get(gk, {}).get('encoder_layer_idx') == d]
                bv = []
                for gk in dec_gks:
                    cs = pg_cos.get(gk, {}).get(pk)
                    if cs:
                        bv.append(cs['mean'])
                vals.append(float(np.mean(bv)) if bv else np.nan)
            ax.plot(enc_indices_valid, vals, 'o-', label=f"{focus_task} vs {aux}",
                    color=TASK_COLORS.get(aux, 'gray'), linewidth=2)
        ax.axhline(0, color='r', linestyle='--', linewidth=0.8)
        ax.set_title(f'Cosine Similarity: {focus_task} vs auxiliary tasks')
        ax.set_xlabel('BEV Encoder Layer')
        ax.set_ylabel('Mean Cosine')
        ax.legend()

        ax = axes[0, 1]
        pg_intf = stats.get('per_group_interference', {})
        for aux in aux_tasks:
            ik = f"{aux}_on_{focus_task}"
            vals = []
            for d in enc_indices_valid:
                dec_gks = [gk for gk in pg_intf if group_meta.get(gk, {}).get('encoder_layer_idx') == d]
                bv = []
                for gk in dec_gks:
                    iv = pg_intf.get(gk, {}).get(ik)
                    if iv:
                        bv.append(iv['mean'])
                vals.append(float(np.mean(bv)) if bv else 0)
            ax.plot(enc_indices_valid, vals, 's-', label=f"{aux} -> {focus_task}",
                    color=TASK_COLORS.get(aux, 'gray'), linewidth=2)
        ax.set_title(f'Interference Magnitude on {focus_task}')
        ax.set_xlabel('BEV Encoder Layer')
        ax.set_ylabel('Interference')
        ax.legend()

        ax = axes[1, 0]
        overall_cos = stats.get('pairwise_cosine', {})
        overall_norms = stats.get('gradient_norms', {})
        for aux in aux_tasks:
            pk = f"{focus_task}_vs_{aux}" if f"{focus_task}_vs_{aux}" in overall_cos else f"{aux}_vs_{focus_task}"
            cs = overall_cos.get(pk, {})
            ns = overall_norms.get(aux, {})
            if cs and ns:
                score = cs['mean'] * ns['mean']
                ax.bar(aux, score, color=TASK_COLORS.get(aux, 'gray'), alpha=0.8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(f'Helpfulness Score: cos(g_aux, g_{focus_task}) * ||g_aux||')
        ax.set_ylabel('Score (positive = helpful)')

        ax = axes[1, 1]
        ax.axis('off')
        lines = [f"=== {focus_task.upper()}-CENTRIC SUMMARY ===\n"]
        for aux in aux_tasks:
            pk = f"{focus_task}_vs_{aux}" if f"{focus_task}_vs_{aux}" in overall_cos else f"{aux}_vs_{focus_task}"
            cs = overall_cos.get(pk, {})
            cf = stats.get('conflict_frequency', {}).get(pk, {})
            intf_key = f"{aux}_on_{focus_task}"
            iv = intf.get(intf_key, {})
            lines.append(f"{aux}:")
            lines.append(f"  cosine = {cs.get('mean', 'N/A'):+.4f}" if isinstance(cs.get('mean'), float) else f"  cosine = N/A")
            lines.append(f"  conflict rate = {cf.get('ratio', 0) * 100:.1f}%")
            lines.append(f"  interference = {iv.get('mean', 0):.4f}" if isinstance(iv.get('mean'), float) else f"  interference = N/A")
            lines.append("")
        ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        plt.suptitle(f'VAD: Planning-Centric Gradient Analysis Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'planning_centric_dashboard.png'), dpi=150)
        plt.close()

    print(f"Advanced visualizations saved to {adv_dir}")


# ============================================================================
# Visualizations — Direction (PCA)
# ============================================================================

def create_direction_visualizations(stats, output_dir, group_meta=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from sklearn.decomposition import PCA
    except ImportError:
        print("Warning: matplotlib or sklearn not installed. Skipping direction visualizations.")
        return

    dir_dir = os.path.join(output_dir, 'direction')
    os.makedirs(dir_dir, exist_ok=True)

    grad_dirs = stats.get('gradient_directions', {})
    if not grad_dirs:
        print("Warning: No gradient directions stored. Use --analysis-mode direction")
        return

    task_names = list(grad_dirs.keys())
    if len(task_names) < 2:
        return

    # Stack all gradients for PCA
    all_grads = []
    all_labels = []
    for task in task_names:
        for g in grad_dirs[task]:
            all_grads.append(g.numpy())
            all_labels.append(task)

    X = np.stack(all_grads)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # --- PCA Gradient Compass ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for task in task_names:
        mask = [l == task for l in all_labels]
        pts = X_2d[mask]
        color = TASK_COLORS.get(task, 'gray')

        ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.3, s=30, label=f'{task} (per batch)')

        mean_pt = pts.mean(axis=0)
        ax.annotate('', xy=mean_pt, xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.text(mean_pt[0], mean_pt[1], f' {task}', fontsize=11, fontweight='bold', color=color)

        if len(pts) > 2:
            cov = np.cov(pts, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            w, h = 2 * 2 * np.sqrt(eigenvalues)
            ellipse = Ellipse(xy=mean_pt, width=w, height=h, angle=angle,
                              edgecolor=color, facecolor=color, alpha=0.1, linewidth=1.5)
            ax.add_patch(ellipse)

    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    ax.set_title('VAD: Task Gradient Direction Compass (PCA 2D Projection)')
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_dir, 'pca_gradient_compass.png'), dpi=150)
    plt.close()

    # --- Cosine Stability Time Series ---
    num_batches = len(grad_dirs[task_names[0]])
    fig, ax = plt.subplots(figsize=(12, 5))

    for t1, t2 in TASK_PAIRS:
        if t1 not in grad_dirs or t2 not in grad_dirs:
            continue
        cosines = []
        for bi in range(num_batches):
            g1 = grad_dirs[t1][bi]
            g2 = grad_dirs[t2][bi]
            cosines.append(compute_cosine_similarity(g1, g2))

        cosines = np.array(cosines)
        pk = f"{t1}_vs_{t2}"
        ax.plot(range(num_batches), cosines, alpha=0.5, linewidth=1, label=pk)

        if len(cosines) > 5:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(cosines, kernel, mode='valid')
            ax.plot(range(2, 2 + len(smoothed)), smoothed, linewidth=2.5)

    ax.axhline(0, color='r', linestyle='--', linewidth=0.8, label='Conflict threshold')
    ax.set_xlabel('Batch Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('VAD: Gradient Cosine Stability Across Batches')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_dir, 'cosine_stability_timeseries.png'), dpi=150)
    plt.close()

    # --- Angular Histogram ---
    n_pairs = len(TASK_PAIRS)
    cols = min(3, n_pairs)
    rows = math.ceil(n_pairs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), subplot_kw={'projection': 'polar'})
    axes_flat = np.array(axes).flatten() if n_pairs > 1 else [axes]

    for idx, (t1, t2) in enumerate(TASK_PAIRS):
        if t1 not in grad_dirs or t2 not in grad_dirs:
            continue
        angles = []
        for bi in range(num_batches):
            g1 = grad_dirs[t1][bi]
            g2 = grad_dirs[t2][bi]
            cos = compute_cosine_similarity(g1, g2)
            if not np.isnan(cos):
                angles.append(np.arccos(np.clip(cos, -1, 1)))

        if not angles:
            continue

        ax = axes_flat[idx]
        angles = np.array(angles)
        bins = np.linspace(0, np.pi, 19)
        counts, _ = np.histogram(angles, bins=bins)

        theta = (bins[:-1] + bins[1:]) / 2
        width = bins[1] - bins[0]
        colors = ['#2ecc71' if t < np.pi / 2 else '#e74c3c' for t in theta]
        ax.bar(theta, counts, width=width, color=colors, alpha=0.7)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_title(f'{t1} vs {t2}', fontsize=10, pad=15)
        ax.axvline(np.pi / 2, color='gray', linestyle='--', linewidth=0.8)

    for idx in range(len(TASK_PAIRS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('VAD: Gradient Angle Distribution (green < 90° = cooperative, red > 90° = conflict)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_dir, 'angular_histogram.png'), dpi=150)
    plt.close()

    print(f"Direction visualizations saved to {dir_dir}")


# ============================================================================
# Random Baseline & Statistical Significance
# ============================================================================

def generate_random_baseline(dim: int, num_pairs: int = 10000, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    cosines = []
    for _ in range(num_pairs):
        v1 = rng.randn(dim)
        v2 = rng.randn(dim)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-10 and n2 > 1e-10:
            cosines.append(np.dot(v1, v2) / (n1 * n2))
    return np.array(cosines)


def compute_statistical_tests(observed_cosines: np.ndarray, random_cosines: np.ndarray) -> dict:
    obs = observed_cosines[~np.isnan(observed_cosines)]
    rand = random_cosines[~np.isnan(random_cosines)]

    if len(obs) < 2 or len(rand) < 2:
        return {'error': 'insufficient data'}

    t_stat, t_pval = sp_stats.ttest_ind(obs, rand, equal_var=False)
    ks_stat, ks_pval = sp_stats.ks_2samp(obs, rand)

    pooled_std = np.sqrt((np.var(obs) + np.var(rand)) / 2)
    cohens_d = (np.mean(obs) - np.mean(rand)) / pooled_std if pooled_std > 1e-10 else 0.0

    return {
        't_test': {'statistic': float(t_stat), 'p_value': float(t_pval)},
        'ks_test': {'statistic': float(ks_stat), 'p_value': float(ks_pval)},
        'effect_size_cohens_d': float(cohens_d),
        'observed': {
            'mean': float(np.mean(obs)), 'std': float(np.std(obs)),
            'median': float(np.median(obs)), 'n': len(obs),
        },
        'random_baseline': {
            'mean': float(np.mean(rand)), 'std': float(np.std(rand)),
            'median': float(np.median(rand)), 'n': len(rand),
        },
    }


def detect_bimodality(values: np.ndarray) -> dict:
    vals = values[~np.isnan(values)]
    if len(vals) < 10:
        return {'error': 'insufficient data'}

    excess_kurtosis = float(sp_stats.kurtosis(vals, fisher=True))
    skewness = float(sp_stats.skew(vals))
    neg_frac = float(np.mean(vals < 0))
    pos_frac = float(np.mean(vals > 0))
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    spread_ratio = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('inf')

    is_bimodal_kurtosis = excess_kurtosis < -1.0
    is_bimodal_balance = 0.25 < neg_frac < 0.75
    bimodality_score = 0.0
    if is_bimodal_kurtosis:
        bimodality_score += 0.4
    if is_bimodal_balance and spread_ratio > 10:
        bimodality_score += 0.3

    return {
        'excess_kurtosis': excess_kurtosis,
        'skewness': skewness,
        'negative_fraction': neg_frac,
        'positive_fraction': pos_frac,
        'mean': mean_val,
        'std': std_val,
        'spread_ratio': spread_ratio,
        'is_bimodal_kurtosis': is_bimodal_kurtosis,
        'bimodality_score': bimodality_score,
        'interpretation': (
            'BIMODAL (strong conflict + cooperation alternating)'
            if bimodality_score >= 0.6
            else 'POSSIBLY BIMODAL (mixed signals)'
            if bimodality_score >= 0.3
            else 'UNIMODAL (consistent direction)'
        ),
    }


def run_random_baseline_analysis(stats: dict, output_dir: str) -> dict:
    baseline_results = {
        'statistical_tests': {},
        'bimodality': {},
    }

    pg_norms = stats.get('per_group_norms', {})
    total_params = len(pg_norms) * 1000 if pg_norms else 100000

    cos_data = stats.get('pairwise_cosine', {})

    print("\nGenerating random baseline for statistical comparison...")
    random_cosines = generate_random_baseline(total_params, num_pairs=10000)
    baseline_results['random_baseline_dim'] = total_params
    baseline_results['random_baseline_cosines'] = {
        'mean': float(np.mean(random_cosines)),
        'std': float(np.std(random_cosines)),
    }

    for pk, cs in cos_data.items():
        vals = cs.get('values', [])
        if not vals:
            continue
        obs = np.array(vals)
        baseline_results['statistical_tests'][pk] = compute_statistical_tests(obs, random_cosines)
        baseline_results['bimodality'][pk] = detect_bimodality(obs)

    os.makedirs(output_dir, exist_ok=True)
    baseline_dir = os.path.join(output_dir, 'baseline_comparison')
    os.makedirs(baseline_dir, exist_ok=True)

    with open(os.path.join(baseline_dir, 'statistical_tests.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("RANDOM BASELINE COMPARISON & BIMODALITY ANALYSIS")
    print("=" * 70)
    print(f"\nRandom baseline: dim={total_params}, mean={np.mean(random_cosines):.6f}, std={np.std(random_cosines):.6f}")

    for pk in sorted(baseline_results['statistical_tests'].keys()):
        st = baseline_results['statistical_tests'][pk]
        bm = baseline_results['bimodality'].get(pk, {})
        print(f"\n  {pk}:")
        if 'error' not in st:
            sig = "***" if st['t_test']['p_value'] < 0.001 else "**" if st['t_test']['p_value'] < 0.01 else "*" if st['t_test']['p_value'] < 0.05 else "n.s."
            print(f"    Observed: mean={st['observed']['mean']:+.6f}, std={st['observed']['std']:.6f} (n={st['observed']['n']})")
            print(f"    t-test vs random: t={st['t_test']['statistic']:.3f}, p={st['t_test']['p_value']:.4f} [{sig}]")
            print(f"    KS-test vs random: D={st['ks_test']['statistic']:.3f}, p={st['ks_test']['p_value']:.4f}")
            print(f"    Effect size (Cohen's d): {st['effect_size_cohens_d']:.4f}")
        if 'error' not in bm:
            print(f"    Bimodality: kurtosis={bm['excess_kurtosis']:.3f}")
            print(f"    Distribution: {bm['interpretation']}")
            print(f"    Neg/Pos split: {bm['negative_fraction']*100:.1f}% / {bm['positive_fraction']*100:.1f}%")

    print("=" * 70)
    return baseline_results


def create_baseline_visualizations(stats: dict, baseline_results: dict, output_dir: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping baseline visualizations.")
        return

    baseline_dir = os.path.join(output_dir, 'baseline_comparison')
    os.makedirs(baseline_dir, exist_ok=True)

    cos_data = stats.get('pairwise_cosine', {})
    pairs = sorted(cos_data.keys())
    if not pairs:
        return

    dim = baseline_results.get('random_baseline_dim', 100000)
    random_cosines = generate_random_baseline(dim, num_pairs=10000)

    # KDE + Histogram per task pair
    n_pairs = len(pairs)
    cols = min(3, n_pairs)
    rows = math.ceil(n_pairs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_flat = np.array(axes).flatten() if n_pairs > 1 else [axes]

    for idx, pk in enumerate(pairs):
        ax = axes_flat[idx]
        vals = np.array(cos_data[pk].get('values', []))
        if len(vals) == 0:
            continue

        ax.hist(vals, bins=30, density=True, alpha=0.5, color='steelblue', label='Observed', edgecolor='white')

        if len(vals) > 3:
            kde = sp_stats.gaussian_kde(vals)
            x_range = np.linspace(min(vals.min(), -0.2), max(vals.max(), 0.2), 200)
            ax.plot(x_range, kde(x_range), color='steelblue', linewidth=2, label='Observed KDE')

        rand_kde = sp_stats.gaussian_kde(random_cosines)
        x_range_rand = np.linspace(-0.15, 0.15, 200)
        ax.plot(x_range_rand, rand_kde(x_range_rand), color='red', linewidth=2, linestyle='--', label='Random baseline')

        bm = baseline_results.get('bimodality', {}).get(pk, {})
        st = baseline_results.get('statistical_tests', {}).get(pk, {})

        title = pk
        if 'error' not in bm:
            title += f"\n{bm['interpretation']}"
        if 'error' not in st:
            p_val = st['t_test']['p_value']
            sig = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            title += f" | t-test: {sig}"

        ax.set_title(title, fontsize=9)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('VAD: Cosine Similarity Distribution vs Random Baseline', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(baseline_dir, 'cosine_distribution_vs_baseline.png'), dpi=150)
    plt.close()

    # Summary comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    means = [cos_data[pk]['mean'] for pk in pairs]
    stds = [cos_data[pk]['std'] for pk in pairs]
    x = np.arange(len(pairs))
    axes[0].bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7, label='Observed')
    axes[0].axhline(np.mean(random_cosines), color='red', linestyle='--', label=f'Random mean ({np.mean(random_cosines):.4f})')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Mean Cosine Similarity')
    axes[0].set_title('Observed vs Random Baseline')
    axes[0].legend(fontsize=8)

    p_vals = []
    for pk in pairs:
        st = baseline_results.get('statistical_tests', {}).get(pk, {})
        p_vals.append(st.get('t_test', {}).get('p_value', 1.0))
    colors = ['#e74c3c' if p < 0.05 else '#f39c12' if p < 0.1 else '#2ecc71' for p in p_vals]
    axes[1].bar(x, [-np.log10(max(p, 1e-10)) for p in p_vals], color=colors, alpha=0.7)
    axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    axes[1].axhline(-np.log10(0.01), color='darkred', linestyle=':', label='p=0.01')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('-log10(p-value)')
    axes[1].set_title('Statistical Significance (t-test)')
    axes[1].legend(fontsize=8)

    bm_scores = []
    for pk in pairs:
        bm = baseline_results.get('bimodality', {}).get(pk, {})
        bm_scores.append(bm.get('bimodality_score', 0))
    bm_colors = ['#e74c3c' if s >= 0.6 else '#f39c12' if s >= 0.3 else '#2ecc71' for s in bm_scores]
    axes[2].bar(x, bm_scores, color=bm_colors, alpha=0.7)
    axes[2].axhline(0.6, color='red', linestyle='--', label='Bimodal threshold')
    axes[2].axhline(0.3, color='orange', linestyle=':', label='Possibly bimodal')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel('Bimodality Score')
    axes[2].set_title('Distribution Shape Analysis')
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8)

    plt.suptitle('VAD: Gradient Conflict Statistical Validation Summary', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(baseline_dir, 'statistical_summary.png'), dpi=150)
    plt.close()

    print(f"Baseline comparison visualizations saved to {baseline_dir}")


# ============================================================================
# Multi-Checkpoint Epoch-wise Dynamics
# ============================================================================

def analyze_checkpoints(
    config_path: str,
    checkpoint_paths: List[str],
    checkpoint_labels: List[str],
    num_batches: int,
    batch_size: int,
    device: str,
    shared_layers: List[str],
    analysis_target: str = 'bev_encoder',
    fp16: bool = False,
    output_dir: str = 'gradient_dynamics',
    seed: int = 42,
):
    os.chdir(PROJECT_ROOT)

    checkpoint_stats = []

    for ckpt_idx, (ckpt_path, label) in enumerate(zip(checkpoint_paths, checkpoint_labels)):
        print(f"\n{'='*60}")
        print(f"Analyzing checkpoint [{ckpt_idx+1}/{len(checkpoint_paths)}]: {label}")
        print(f"  Path: {ckpt_path}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = Config.fromfile(config_path)
        cfg.data.samples_per_gpu = batch_size
        cfg.data.workers_per_gpu = min(4, batch_size)

        if ckpt_idx == 0:
            model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
            model.init_weights()
            device_id = int(device.split(':')[1]) if ':' in device else 0
            model = model.to(device)
            model = MMDataParallel(model, device_ids=[device_id])
            dataset = custom_build_dataset(cfg.data.train)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=min(4, batch_size),
                collate_fn=partial(collate, samples_per_gpu=batch_size), drop_last=True,
            )

        load_checkpoint(model.module, ckpt_path, map_location='cpu')
        param_groups, group_meta = get_shared_parameters_grouped(model, shared_layers, analysis_target)

        torch.manual_seed(seed)
        np.random.seed(seed)

        all_results = []
        data_iter = iter(dataloader)
        for batch_idx in range(num_batches):
            try:
                data = next(data_iter)
            except StopIteration:
                break
            print(f"  Batch {batch_idx+1}/{num_batches}...", end='\r')
            try:
                result = analyze_single_batch(
                    model, data, param_groups, group_meta, device,
                    fp16=fp16, analysis_mode='full',
                    store_gradients=False, selective_eval=True,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n  Warning: batch {batch_idx} failed: {e}")
                continue
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

        if all_results:
            ckpt_stat = aggregate_results(all_results, analysis_mode='full')
            ckpt_stat['checkpoint_label'] = label
            ckpt_stat['checkpoint_path'] = ckpt_path
            checkpoint_stats.append(ckpt_stat)
            print(f"\n  Processed {len(all_results)} batches for {label}")

    # Save and visualize dynamics
    dynamics_dir = os.path.join(output_dir, 'epoch_dynamics')
    os.makedirs(dynamics_dir, exist_ok=True)

    save_data = []
    for cs in checkpoint_stats:
        entry = {
            'label': cs['checkpoint_label'],
            'pairwise_cosine': {pk: {'mean': v['mean'], 'std': v['std']}
                                 for pk, v in cs.get('pairwise_cosine', {}).items()},
            'conflict_frequency': cs.get('conflict_frequency', {}),
            'gradient_norms': {t: {'mean': v['mean'], 'std': v['std']}
                               for t, v in cs.get('gradient_norms', {}).items()},
        }
        save_data.append(entry)

    with open(os.path.join(dynamics_dir, 'epoch_dynamics.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    create_epoch_dynamics_visualizations(checkpoint_stats, dynamics_dir)

    return checkpoint_stats


def create_epoch_dynamics_visualizations(checkpoint_stats: list, output_dir: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed.")
        return

    if len(checkpoint_stats) < 2:
        print("Need at least 2 checkpoints for dynamics visualization.")
        return

    labels = [cs['checkpoint_label'] for cs in checkpoint_stats]
    x = np.arange(len(labels))

    # Cosine similarity evolution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    pair_keys = sorted(checkpoint_stats[0].get('pairwise_cosine', {}).keys())
    for pk in pair_keys:
        means = []
        for cs in checkpoint_stats:
            v = cs.get('pairwise_cosine', {}).get(pk, {})
            means.append(v.get('mean', np.nan) if isinstance(v, dict) else np.nan)
        ax.plot(x, means, 'o-', label=pk, linewidth=2, markersize=6)

    ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('VAD: Gradient Cosine Evolution Across Training')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for pk in pair_keys:
        ratios = []
        for cs in checkpoint_stats:
            cf = cs.get('conflict_frequency', {}).get(pk, {})
            ratios.append(cf.get('ratio', np.nan) if isinstance(cf, dict) else np.nan)
        ax.plot(x, ratios, 's-', label=pk, linewidth=2, markersize=6)

    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Random baseline (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Conflict Frequency')
    ax.set_title('VAD: Conflict Frequency Evolution')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.suptitle('VAD: Gradient Dynamics Across Training Checkpoints', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(output_dir, 'cosine_evolution.png'), dpi=150)
    plt.close()

    # Gradient norm evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    task_names = sorted(checkpoint_stats[0].get('gradient_norms', {}).keys())
    for task in task_names:
        means = []
        stds = []
        for cs in checkpoint_stats:
            ns = cs.get('gradient_norms', {}).get(task, {})
            means.append(ns.get('mean', 0) if isinstance(ns, dict) else 0)
            stds.append(ns.get('std', 0) if isinstance(ns, dict) else 0)
        color = TASK_COLORS.get(task, 'gray')
        ax.plot(x, means, 'o-', label=task, color=color, linewidth=2, markersize=6)
        ax.fill_between(x, np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.15, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Gradient Norm')
    ax.set_title('VAD: Task Gradient Magnitude Evolution (shaded = ±1σ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_norm_evolution.png'), dpi=150)
    plt.close()

    # Magnitude ratio evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'plan' in task_names:
        for task in task_names:
            if task == 'plan':
                continue
            ratios = []
            for cs in checkpoint_stats:
                n_task = cs.get('gradient_norms', {}).get(task, {}).get('mean', 1)
                n_plan = cs.get('gradient_norms', {}).get('plan', {}).get('mean', 1)
                ratios.append(n_task / max(n_plan, 1e-10))
            color = TASK_COLORS.get(task, 'gray')
            ax.plot(x, ratios, 'o-', label=f'{task}/plan', color=color, linewidth=2, markersize=6)

        ax.axhline(1, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Norm Ratio (task / plan)')
        ax.set_title('VAD: Gradient Magnitude Imbalance vs Plan Task')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'magnitude_ratio_evolution.png'), dpi=150)
    plt.close()

    print(f"Epoch dynamics visualizations saved to {output_dir}")


# ============================================================================
# Recommendations
# ============================================================================

def generate_recommendations(stats, output_dir, focus_task='plan', baseline_results=None):
    lines = []
    lines.append("=" * 70)
    lines.append("VAD GRADIENT CONFLICT ANALYSIS - RECOMMENDATIONS")
    lines.append("=" * 70)
    lines.append("")

    if baseline_results and 'statistical_tests' in baseline_results:
        lines.append("0. STATISTICAL VALIDATION vs RANDOM BASELINE")
        lines.append("-" * 40)
        any_significant = False
        any_bimodal = False
        for pk in sorted(baseline_results['statistical_tests'].keys()):
            st = baseline_results['statistical_tests'][pk]
            bm = baseline_results.get('bimodality', {}).get(pk, {})
            if 'error' in st:
                continue
            p_val = st['t_test']['p_value']
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            if p_val < 0.05:
                any_significant = True
            lines.append(f"  {pk}: t-test p={p_val:.4f} [{sig}], Cohen's d={st['effect_size_cohens_d']:.4f}")
            if 'error' not in bm:
                interp = bm['interpretation']
                lines.append(f"    Distribution: {interp} (kurtosis={bm['excess_kurtosis']:.3f})")
                if bm['bimodality_score'] >= 0.3:
                    any_bimodal = True
        lines.append("")
        if not any_significant:
            lines.append("  CRITICAL FINDING: No task pair shows statistically significant")
            lines.append("  difference from random baseline.")
        elif any_bimodal:
            lines.append("  FINDING: Some task pairs show bimodal distribution.")
            lines.append("  → Gradient surgery (PCGrad/CAGrad) may be justified.")
        else:
            lines.append("  FINDING: Statistically significant but unimodal.")
        lines.append("")

    lines.append("1. OVERALL CONFLICT ASSESSMENT")
    lines.append("-" * 40)
    cf = stats.get('conflict_frequency', {})
    high_conflict = [(k, v['ratio']) for k, v in cf.items() if v['ratio'] > 0.3]
    if high_conflict:
        for pair, ratio in sorted(high_conflict, key=lambda x: -x[1]):
            lines.append(f"  HIGH CONFLICT: {pair} (conflict rate: {ratio * 100:.1f}%)")
    else:
        lines.append("  No significant overall gradient conflicts detected (all < 30%).")
    lines.append("")

    lines.append("2. GRADIENT MAGNITUDE IMBALANCE")
    lines.append("-" * 40)
    norms = stats.get('gradient_norms', {})
    if focus_task in norms:
        fn = norms[focus_task]['mean']
        for t, ns in sorted(norms.items(), key=lambda x: -x[1]['mean']):
            ratio = ns['mean'] / fn if fn > 1e-10 else float('inf')
            marker = " << DOMINATES" if ratio > 5 else ""
            lines.append(f"  {t:10s}: norm={ns['mean']:.4f}  (ratio to {focus_task}: {ratio:.1f}x){marker}")
        any_imbalance = any(ns['mean'] / fn > 5 for t, ns in norms.items() if t != focus_task and fn > 1e-10)
        if any_imbalance:
            lines.append(f"\n  RECOMMENDATION: Consider gradient norm balancing (GradNorm, IMTL)")
            lines.append(f"  to prevent {focus_task} gradient from being dominated.")
    lines.append("")

    lines.append(f"3. INTERFERENCE ON {focus_task.upper()}")
    lines.append("-" * 40)
    intf = stats.get('interference', {})
    if intf:
        for t in ['det', 'map', 'motion']:
            key = f"{t}_on_{focus_task}"
            if key in intf:
                v = intf[key]['mean']
                severity = "HIGH" if v > 1.0 else "MODERATE" if v > 0.3 else "LOW"
                lines.append(f"  {t:10s} -> {focus_task}: {v:.4f} ({severity})")
    lines.append("")

    lines.append("4. PER-LAYER CONFLICT HOTSPOTS")
    lines.append("-" * 40)
    pg_cos = stats.get('per_group_cosine', {})
    pg_cf = stats.get('per_group_conflict_frequency', {})
    if pg_cos:
        all_entries = []
        for gk, pks in pg_cos.items():
            for pk, cs in pks.items():
                cf_ratio = pg_cf.get(gk, {}).get(pk, {}).get('ratio', 0)
                all_entries.append((gk, pk, cs['mean'], cf_ratio))
        all_entries.sort(key=lambda x: x[2])

        lines.append("  Top 15 most conflicting (layer, pair):")
        for gk, pk, mean_cos, cf_ratio in all_entries[:15]:
            lines.append(f"    {gk:30s} | {pk:20s} | cos={mean_cos:+.4f} | conflict={cf_ratio * 100:.1f}%")
    lines.append("")

    lines.append("5. SUGGESTED ACTIONS")
    lines.append("-" * 40)
    suggestions = []
    if high_conflict:
        suggestions.append("- Apply PCGrad or CAGrad to resolve directional conflicts.")
        if pg_cos:
            conflict_layers = set()
            for gk, pks in pg_cf.items():
                for pk, v in pks.items():
                    if v['ratio'] > 0.4:
                        conflict_layers.add(gk)
            if conflict_layers:
                suggestions.append(f"- Focus gradient surgery on: {', '.join(sorted(conflict_layers))}")

    any_imbalance = False
    if focus_task in norms:
        fn = norms[focus_task]['mean']
        any_imbalance = any(ns['mean'] / fn > 5 for t, ns in norms.items() if t != focus_task and fn > 1e-10)
    if any_imbalance:
        suggestions.append(f"- Apply GradNorm or dynamic loss weighting to balance {focus_task} signal.")

    if not suggestions:
        suggestions.append("- No critical issues detected. Current training may be adequate.")
        suggestions.append("- Monitor planning metrics to confirm.")

    for s in suggestions:
        lines.append(f"  {s}")
    lines.append("")

    text = '\n'.join(lines)
    filepath = os.path.join(output_dir, 'recommendations.txt')
    with open(filepath, 'w') as f:
        f.write(text)
    print(f"Recommendations saved to {filepath}")
    return text


# ============================================================================
# Debug Mode
# ============================================================================

def debug_mode(model, dataloader, device, fp16=False):
    print("\n" + "=" * 60)
    print("DEBUG MODE - Inspecting VAD model and loss structure")
    print("=" * 60)

    raw_model = model.module if hasattr(model, 'module') else model

    # BEV Encoder structure
    encoder = raw_model.pts_bbox_head.transformer.encoder
    print(f"\nBEV Encoder: {type(encoder).__name__}")
    print(f"  Number of layers: {len(encoder.layers)}")
    for i, layer in enumerate(encoder.layers):
        print(f"  Layer {i}: {type(layer).__name__}")
        if hasattr(layer, 'operation_order'):
            print(f"    operation_order: {layer.operation_order}")
        if hasattr(layer, 'attentions'):
            for j, attn in enumerate(layer.attentions):
                print(f"    attentions[{j}]: {type(attn).__name__}")
        if hasattr(layer, 'ffns'):
            for j, ffn in enumerate(layer.ffns):
                print(f"    ffns[{j}]: {type(ffn).__name__}")
        if hasattr(layer, 'norms'):
            print(f"    norms: {len(layer.norms)} LayerNorm modules")

    # Decoder structure
    transformer = raw_model.pts_bbox_head.transformer
    print(f"\nDetection Decoder: {type(transformer.decoder).__name__} ({len(transformer.decoder.layers)} layers)")
    print(f"Map Decoder: {type(transformer.map_decoder).__name__} ({len(transformer.map_decoder.layers)} layers)")

    # Head decoders
    head = raw_model.pts_bbox_head
    for dec_name in ['motion_decoder', 'motion_map_decoder', 'ego_agent_decoder', 'ego_map_decoder']:
        dec = getattr(head, dec_name, None)
        if dec is not None:
            print(f"{dec_name}: {type(dec).__name__} ({len(dec.layers)} layers)")

    # Forward pass to get loss keys
    print("\n--- Running forward pass to collect loss keys ---")
    data = next(iter(dataloader))

    model.train()
    with torch.cuda.amp.autocast(enabled=fp16):
        outputs = model(**data)

    print(f"\nLoss keys returned by model ({len(outputs)} total):")
    for key, val in sorted(outputs.items()):
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, requires_grad={val.requires_grad}, value={val.item():.6f}")
        else:
            print(f"  {key}: {type(val)}")

    print("\n" + "-" * 40)
    print("Task groups mapping:")
    for task, prefixes in TASK_GROUPS.items():
        matched = [k for k in outputs.keys() if match_loss_key(k, prefixes)]
        print(f"  {task}: {len(matched)} losses matched")
        for m in matched:
            print(f"    - {m}")

    print("=" * 60 + "\n")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze gradient conflicts in VAD')
    parser.add_argument('--config', type=str,
                        default='projects/configs/VAD/VAD_tiny_e2e.py',
                        help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint file path (required for single-checkpoint mode)')
    parser.add_argument('--num-batches', type=int, default=64,
                        help='Number of batches to analyze (default: 64 = 1 weight update with 8 GPUs × 8 batch)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per GPU (default: 1; 64 batches × 1 = 64 samples)')
    parser.add_argument('--output-dir', type=str, default='gradient_analysis_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use FP16 (default: False, VAD tiny does not use fp16)')
    parser.add_argument('--shared-layers', type=str, nargs='+',
                        default=['temporal_self_attn', 'spatial_cross_attn', 'ffn', 'norm'],
                        help='Names of shared layer types to analyze in BEV encoder')
    parser.add_argument('--analysis-target', type=str, default='bev_encoder',
                        choices=['bev_encoder', 'backbone', 'all'],
                        help='Which shared parameters to analyze '
                             '(bev_encoder: BEV encoder only, backbone: ResNet+FPN, all: both)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to print loss keys and exit')
    parser.add_argument('--analysis-mode', type=str, default='full',
                        choices=['basic', 'full', 'direction'],
                        help='basic: overall only; full: +per-layer+advanced; direction: +PCA')
    parser.add_argument('--focus-task', type=str, default='plan',
                        help='Task to focus planning-centric analysis on')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip visualization, only save numerical results')
    parser.add_argument('--no-selective-eval', action='store_true',
                        help='Disable selective eval')
    parser.add_argument('--random-baseline', action='store_true',
                        help='Run random baseline comparison with statistical tests')
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None,
                        help='Multiple checkpoint paths for epoch-wise dynamics analysis')
    parser.add_argument('--checkpoint-labels', type=str, nargs='+', default=None,
                        help='Labels for each checkpoint (e.g., ep1 ep10 ep20)')
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if not args.checkpoint and not args.checkpoints:
        print("Error: either --checkpoint or --checkpoints is required.")
        sys.exit(1)

    # Multi-checkpoint mode
    if args.checkpoints:
        ckpt_paths = args.checkpoints
        ckpt_labels = args.checkpoint_labels or [f"ckpt_{i}" for i in range(len(ckpt_paths))]
        if len(ckpt_labels) != len(ckpt_paths):
            print(f"Warning: {len(ckpt_labels)} labels for {len(ckpt_paths)} checkpoints. Padding.")
            ckpt_labels = ckpt_labels + [f"ckpt_{i}" for i in range(len(ckpt_labels), len(ckpt_paths))]

        analyze_checkpoints(
            config_path=args.config,
            checkpoint_paths=ckpt_paths,
            checkpoint_labels=ckpt_labels,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            device=args.device,
            shared_layers=args.shared_layers,
            analysis_target=args.analysis_target,
            fp16=args.fp16,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(f"\nMulti-checkpoint analysis complete! Results saved to: {args.output_dir}/epoch_dynamics/")
        return

    store_gradients = (args.analysis_mode == 'direction')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.chdir(PROJECT_ROOT)

    cfg = Config.fromfile(args.config)
    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.workers_per_gpu = min(4, args.batch_size)

    print(f"Loading config from: {args.config}")
    print(f"Loading checkpoint from: {args.checkpoint}")
    print(f"Analysis mode: {args.analysis_mode}")
    print(f"Analysis target: {args.analysis_target}")
    print(f"FP16: {args.fp16}")
    print(f"Analyzing {args.num_batches} batches × batch_size {args.batch_size} "
          f"= {args.num_batches * args.batch_size} samples")

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print("Checkpoint loaded successfully")

    device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
    model = model.to(args.device)
    model = MMDataParallel(model, device_ids=[device_id])

    # Get grouped shared parameters
    param_groups, group_meta = get_shared_parameters_grouped(
        model, args.shared_layers, args.analysis_target
    )

    total_groups = len(param_groups)
    total_params = sum(p.numel() for grp in param_groups.values() for p in grp.values())
    print(f"\nFound {total_groups} parameter groups ({total_params:,} parameters):")
    for gk, params in param_groups.items():
        n = sum(p.numel() for p in params.values())
        meta = group_meta.get(gk, {})
        print(f"  {gk:35s}: {n:>8,} params  (layer={meta.get('encoder_layer_idx', 'N/A')}, op={meta.get('op_type', 'N/A')})")

    if total_params == 0:
        print("\nWarning: No shared parameters found! Check --shared-layers and --analysis-target arguments.")
        return

    dataset = custom_build_dataset(cfg.data.train)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=min(4, args.batch_size),
        collate_fn=partial(collate, samples_per_gpu=args.batch_size), drop_last=True,
    )

    print(f"\nDataset size: {len(dataset)}")

    if args.debug:
        debug_mode(model, dataloader, args.device, args.fp16)
        return

    print(f"\nStarting gradient conflict analysis...\n")

    all_results = []
    for batch_idx, data in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        print(f"Processing batch {batch_idx + 1}/{args.num_batches}...", end='\r')

        try:
            result = analyze_single_batch(
                model, data, param_groups, group_meta, args.device,
                fp16=args.fp16, analysis_mode=args.analysis_mode,
                store_gradients=store_gradients,
                selective_eval=not args.no_selective_eval,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nWarning: Failed to process batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()

    print(f"\nProcessed {len(all_results)} batches successfully")

    if not all_results:
        print("Error: No batches were processed successfully")
        return

    stats = aggregate_results(all_results, analysis_mode=args.analysis_mode)

    print_summary(stats, analysis_mode=args.analysis_mode, focus_task=args.focus_task)
    save_results(stats, args.output_dir, analysis_mode=args.analysis_mode)

    if not args.no_plots:
        try:
            create_basic_visualizations(stats, args.output_dir)
        except Exception as e:
            print(f"Warning: Failed to create basic visualizations: {e}")

        if args.analysis_mode in ('full', 'direction'):
            try:
                create_per_layer_visualizations(stats, group_meta, args.output_dir)
            except Exception as e:
                print(f"Warning: Failed to create per-layer visualizations: {e}")

            try:
                create_advanced_visualizations(stats, group_meta, args.output_dir, focus_task=args.focus_task)
            except Exception as e:
                print(f"Warning: Failed to create advanced visualizations: {e}")

        if args.analysis_mode == 'direction':
            try:
                create_direction_visualizations(stats, args.output_dir, group_meta=group_meta)
            except Exception as e:
                print(f"Warning: Failed to create direction visualizations: {e}")

    # Random baseline comparison
    baseline_results = None
    if args.random_baseline:
        try:
            baseline_results = run_random_baseline_analysis(stats, args.output_dir)
            if not args.no_plots:
                create_baseline_visualizations(stats, baseline_results, args.output_dir)
        except Exception as e:
            print(f"Warning: Failed to run random baseline analysis: {e}")
            import traceback
            traceback.print_exc()

    # Generate recommendations
    if args.analysis_mode in ('full', 'direction'):
        try:
            rec = generate_recommendations(
                stats, args.output_dir, focus_task=args.focus_task,
                baseline_results=baseline_results,
            )
            print("\n" + rec)
        except Exception as e:
            print(f"Warning: Failed to generate recommendations: {e}")

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
