"""Seeded three-dimensional RRT with certified edges and safe shortcutting."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from navigation.scene import VascularScene


@dataclass(frozen=True)
class RRTConfig:
    seed: int = 42
    step_cm: float = 0.08
    goal_bias: float = 0.15
    max_iterations: int = 16000
    edge_resolution_cm: float = 0.015
    shortcut_attempts: int = 150
    path_spacing_cm: float = 0.025


@dataclass(frozen=True)
class Plan:
    path: np.ndarray
    tree_nodes: np.ndarray
    tree_parents: np.ndarray
    length_cm: float
    elapsed_s: float
    iterations: int
    seed: int


class PlanningError(RuntimeError):
    pass


def resample_polyline(path, spacing: float) -> np.ndarray:
    """Subdivide each edge individually, never cut a corner by interpolation."""
    result = [np.asarray(path[0], float)]
    for a, b in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b-a)/spacing)))
        result.extend(np.linspace(a, b, count+1)[1:])
    return np.asarray(result)


def plan_rrt(scene: VascularScene, goal_index: int = 0, cfg: RRTConfig | None = None) -> Plan:
    cfg = cfg or RRTConfig()
    if cfg.step_cm <= 0 or cfg.path_spacing_cm <= 0 or cfg.edge_resolution_cm <= 0:
        raise ValueError('RRT distances must be positive')
    if not 0 <= cfg.goal_bias <= 1 or cfg.max_iterations < 1:
        raise ValueError('Invalid RRT goal bias or iteration budget')
    if not 0 <= goal_index < len(scene.goals):
        raise ValueError('Unknown goal branch')
    start, goal = np.asarray(scene.start, float), np.asarray(scene.goals[goal_index], float)
    if scene.clearance(start) <= 0 or scene.clearance(goal) <= 0:
        raise PlanningError('Start or goal is outside the collision-free lumen')
    clock = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    nodes = np.empty((cfg.max_iterations+2, 3), dtype=float)
    parents = np.full(cfg.max_iterations+2, -1, dtype=int)
    nodes[0] = start
    size, end_index = 1, None
    for iteration in range(1, cfg.max_iterations+1):
        sample = goal if rng.random() < cfg.goal_bias else scene.sample_lumen(rng)
        if scene.clearance(sample) <= 0:
            continue
        nearest = int(np.argmin(np.sum((nodes[:size]-sample)**2, axis=1)))
        direction = sample-nodes[nearest]
        distance = np.linalg.norm(direction)
        if distance < 1e-10:
            continue
        new = nodes[nearest]+direction*min(1., cfg.step_cm/distance)
        if not scene.edge_free(nodes[nearest], new, cfg.edge_resolution_cm):
            continue
        nodes[size], parents[size] = new, nearest
        index = size
        size += 1
        if np.linalg.norm(new-goal) <= 2*cfg.step_cm and scene.edge_free(new, goal, cfg.edge_resolution_cm):
            nodes[size], parents[size] = goal, index
            end_index = size
            size += 1
            break
    if end_index is None:
        raise PlanningError(f'RRT found no feasible path after {cfg.max_iterations} iterations; motion was not started')
    indices = []
    while end_index >= 0:
        indices.append(end_index)
        end_index = int(parents[end_index])
    path = nodes[indices[::-1]].copy()
    for _ in range(cfg.shortcut_attempts):
        if len(path) < 3:
            break
        i, j = np.sort(rng.choice(len(path), size=2, replace=False))
        if j > i+1 and scene.edge_free(path[i], path[j], cfg.edge_resolution_cm):
            path = np.vstack((path[:i+1], path[j:]))
    path = resample_polyline(path, cfg.path_spacing_cm)
    if not all(scene.edge_free(a, b, cfg.edge_resolution_cm) for a, b in zip(path[:-1], path[1:])):
        raise PlanningError('Final path failed swept-volume collision validation')
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    arrays = (path, nodes[:size].copy(), parents[:size].copy())
    for array in arrays:
        array.setflags(write=False)
    return Plan(*arrays, length, time.perf_counter()-clock, iteration, cfg.seed)
