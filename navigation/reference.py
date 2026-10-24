"""Slow RRT polyline following, gated by measured tracking error (cm)."""
from __future__ import annotations

import numpy as np
from navigation.scene import default_scene
from navigation.rrt import RRTConfig, plan_rrt


GOAL_BRANCH = 0  # Later upward side branch; the main trunk is not a goal.
SCENE_KIND = 'vessels'
REFERENCE_SPEED_CM_S = 0.05
MAX_TRACKING_ERROR_CM = 0.06
GOAL_TOLERANCE_CM = 0.06
GOAL_SETTLE_S = 1.0

_cache_key = None
_scene = None
_plan = None
_arc = None
_distance = 0.0
_settled = 0.0
_status = 'ready'


def get_navigation():
    global _cache_key, _scene, _plan, _arc
    cfg = RRTConfig()
    key = (SCENE_KIND, GOAL_BRANCH, cfg)
    if _plan is None or _cache_key != key:
        if SCENE_KIND in ('static_obstacles', 'slalom'):
            from navigation.static_scene import static_scene, plan_curve_rrt
            if SCENE_KIND == 'slalom':
                from navigation.slalom_scene import slalom_scene
                scene = slalom_scene()
            else:
                scene = static_scene()
            plan = plan_curve_rrt(scene, cfg)
        else:
            scene = default_scene()
            plan = plan_rrt(scene, GOAL_BRANCH, cfg)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(plan.path, axis=0), axis=1))]
        _scene, _plan, _arc, _cache_key = scene, plan, arc, key
    return _scene, _plan


def reset_navigation() -> None:
    global _distance, _settled, _status
    get_navigation()  # Planning failure propagates before any motion starts.
    _distance, _settled, _status = 0., 0., 'ready'


def _point_at(distance):
    return np.array([np.interp(distance, _arc, _plan.path[:, i]) for i in range(3)])


def get_navigation_reference(*, advance, dt_s=0., current_position_cm=None):
    global _distance, _settled, _status
    scene, plan = get_navigation()
    target = _point_at(_distance)
    if advance:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('Navigation dt must be finite and nonnegative')
        current = None if current_position_cm is None else np.asarray(current_position_cm, float).reshape(3)
        if current is None or not np.all(np.isfinite(current)):
            _status = 'waiting_for_measurement'
        elif scene.clearance(current) <= 0:
            _status = 'outside_virtual_free_space'
            _settled = 0.
        elif np.linalg.norm(current-target) > MAX_TRACKING_ERROR_CM:
            _status = 'waiting_for_bead'
            _settled = 0.
        else:
            proposal = min(plan.length_cm, _distance+REFERENCE_SPEED_CM_S*dt)
            candidate = _point_at(proposal)
            if scene.edge_free(current, candidate):
                _distance = proposal
                target = candidate
                _status = 'tracking'
            else:
                _status = 'correction_segment_blocked'
            at_goal = (_distance >= plan.length_cm-1e-10
                       and np.linalg.norm(current-plan.path[-1]) <= GOAL_TOLERANCE_CM)
            _settled = _settled+dt if at_goal else 0.
            if _settled >= GOAL_SETTLE_S:
                _status = 'goal_reached'
    return target, plan.path, {
        'mode': 'rrt_navigation', 'navigation_status': _status,
        'progress': float(_distance/plan.length_cm),
        'distance_cm': _distance, 'path_length_cm': plan.length_cm,
        'finished': _status == 'goal_reached',
    }


def navigation_snapshot() -> dict:
    scene, plan = get_navigation()
    return {
        'NAV_GOAL_BRANCH': GOAL_BRANCH,
        'NAV_SCENE_KIND': SCENE_KIND,
        'NAV_REFERENCE_SPEED_CM_S': REFERENCE_SPEED_CM_S,
        'NAV_MAX_TRACKING_ERROR_CM': MAX_TRACKING_ERROR_CM,
        'NAV_GOAL_TOLERANCE_CM': GOAL_TOLERANCE_CM,
        'NAV_GOAL_SETTLE_S': GOAL_SETTLE_S,
        'navigation_scene': scene.snapshot(),
        'rrt_seed': plan.seed, 'rrt_iterations': plan.iterations,
        'rrt_planning_s': plan.elapsed_s, 'rrt_path_length_cm': plan.length_cm,
        'rrt_path_cm': plan.path.copy(), 'rrt_tree_nodes_cm': plan.tree_nodes.copy(),
        'rrt_tree_parents': plan.tree_parents.copy(),
    }
