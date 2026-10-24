"""Sine/cosine guided 3D RRT around static irregular solid obstacles."""
from dataclasses import replace
import time
import numpy as np
from navigation.scene import VascularScene, Obstacle
from navigation.rrt import Plan, PlanningError, plan_rrt


def curve(t):
    t = np.asarray(t)
    return np.stack((.34*np.sin(2*np.pi*t), 1.-2.*t,
                     .58-.18*np.cos(4*np.pi*t)), axis=-1)


def static_scene():
    guide = curve(np.linspace(0., 1., 161))
    obstacles = []
    # Each group is a UNION of overlapping solid spheres. Its lobed boundary
    # is irregular in XYZ; these exact solids are used for rendering/collision.
    for t, shift in ((.31, (.055, 0., .025)), (.70, (-.045, 0., -.025))):
        centre = curve(t)+shift
        for offset, radius in (((0.,0.,0.),.13), ((.09,.035,.045),.09),
                               ((-.07,-.04,.055),.085), ((.015,-.065,-.07),.08)):
            obstacles.append(Obstacle(tuple(centre+offset), radius))
    return VascularScene(
        vessels=(), obstacles=tuple(obstacles), start=tuple(guide[0]),
        goals=(tuple(guide[-1]),), workspace=((-0.85,-1.25,.10),(.85,1.25,1.05)),
        guide=tuple(tuple(p) for p in guide), kind='static_obstacles',
    )


def plan_curve_rrt(scene, cfg):
    """RRT through ordered free curve anchors; blocked anchors are bypassed.

    Do not shortcut across the whole path: that would erase the sine shape.
    Each inter-anchor path and its local simplification use swept-bead checks.
    """
    clock = time.perf_counter()
    guide = np.asarray(scene.guide)
    indices = sorted(set(range(0, len(guide), 8)) | {len(guide)-1})
    anchors = [guide[i] for i in indices if scene.clearance(guide[i]) > .025]
    if len(anchors) < 2 or not np.allclose(anchors[0], scene.start) or not np.allclose(anchors[-1], scene.goals[0]):
        raise PlanningError('Guide start/goal is obstructed')
    paths, nodes, parents = [], [], []
    iterations, count = 0, 0
    for i, (start, goal) in enumerate(zip(anchors[:-1], anchors[1:])):
        subscene = replace(scene, start=tuple(start), goals=(tuple(goal),))
        subplan = plan_rrt(subscene, 0, replace(cfg, seed=cfg.seed+i))
        paths.append(subplan.path if i == 0 else subplan.path[1:])
        nodes.append(subplan.tree_nodes)
        parents.append(np.where(subplan.tree_parents < 0, -1, subplan.tree_parents+count))
        count += len(subplan.tree_nodes)
        iterations += subplan.iterations
    path = np.vstack(paths)
    # Round local corners only when both new chords remain collision-free.
    # Endpoints stay fixed; no unchecked spline may cut through a solid.
    for _ in range(1 if scene.kind == 'slalom' else 6):
        for i in range(1, len(path)-1):
            proposal = .25*path[i-1]+.5*path[i]+.25*path[i+1]
            if scene.edge_free(path[i-1], proposal, cfg.edge_resolution_cm) and scene.edge_free(proposal, path[i+1], cfg.edge_resolution_cm):
                path[i] = proposal
    tree, parent = np.vstack(nodes), np.concatenate(parents)
    for array in (path, tree, parent):
        array.setflags(write=False)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return Plan(path, tree, parent, length, time.perf_counter()-clock, iterations, cfg.seed)
