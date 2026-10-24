"""A virtual vessel lumen: a union of capsules minus spherical obstacles.

All coordinates/radii are centimetres in the magnetic model's XYZ frame.
The physical bead is represented by a sphere, including a safety margin.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Shared world dimensions for planning and all camera projections (cm).
MAIN_HEIGHT_CM = 0.45
SCENE_RADIUS_SCALE = 0.64
SCENE_OVERALL_SCALE = 1.0
SCENE_MIRROR_Y = True  # Mirror left/right in both camera projections.
CLOT_WALL_OFFSET_CM = 0.095
CLOT_RADIUS_CM = 0.035


@dataclass(frozen=True)
class Vessel:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float = 0.22


@dataclass(frozen=True)
class Obstacle:
    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class VesselBranch:
    name: str
    points: tuple[tuple[float, float, float], ...]
    radii: tuple[float, ...]


@dataclass(frozen=True)
class VascularScene:
    vessels: tuple[Vessel, ...]
    obstacles: tuple[Obstacle, ...]
    start: tuple[float, float, float]
    goals: tuple[tuple[float, float, float], ...]
    bead_radius: float = 0.05
    safety_margin: float = 0.02
    branches: tuple[VesselBranch, ...] = ()
    workspace: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    guide: tuple[tuple[float, float, float], ...] = ()
    kind: str = 'vessels'

    @property
    def bounds(self) -> np.ndarray:
        if self.workspace is not None:
            return np.asarray(self.workspace, float)
        starts = np.asarray([v.start for v in self.vessels])
        ends = np.asarray([v.end for v in self.vessels])
        radii = np.asarray([v.radius for v in self.vessels])[:, None]
        return np.array([np.minimum(starts-radii, ends-radii).min(axis=0),
                         np.maximum(starts+radii, ends+radii).max(axis=0)])

    def clearance(self, points) -> np.ndarray:
        """Conservative signed clearance of the bead's centre, after inflation."""
        p = np.asarray(points, dtype=float)
        if self.workspace is not None:
            low, high = np.asarray(self.workspace)
            free = np.minimum(p-low, high-p).min(axis=-1)
        else:
            a = np.asarray([v.start for v in self.vessels])
            b = np.asarray([v.end for v in self.vessels])
            ab = b-a
            t = np.clip(np.sum((p[..., None, :]-a)*ab, axis=-1)
                        / np.sum(ab*ab, axis=-1), 0., 1.)
            distance = np.linalg.norm(p[..., None, :]-(a+t[..., None]*ab), axis=-1)
            free = np.max(np.asarray([v.radius for v in self.vessels])-distance, axis=-1)
        for obstacle in self.obstacles:
            free = np.minimum(free, np.linalg.norm(p-obstacle.center, axis=-1)-obstacle.radius)
        result = free-self.bead_radius-self.safety_margin
        return np.where(np.all(np.isfinite(p), axis=-1), result, -np.inf)

    def edge_free(self, start, end, resolution: float = 0.015) -> bool:
        """Check the swept bead, including the space BETWEEN sampled points.

        Clearance is 1-Lipschitz. A spacing/2 buffer at every sample certifies
        the entire segment rather than accepting thin obstacles between nodes.
        """
        a, b = np.asarray(start, float), np.asarray(end, float)
        distance = float(np.linalg.norm(b-a))
        if not np.isfinite(distance) or resolution <= 0:
            return False
        if distance < 1e-12:
            return bool(self.clearance(a) > 0.)
        intervals = max(1, int(np.ceil(distance/resolution)))
        # A coarse certificate can be inconclusive close to a wall. Refine
        # without weakening the clearance requirement or accepting collisions.
        for _ in range(6):
            samples = np.linspace(a, b, intervals+1)
            clearance = self.clearance(samples)
            if np.any(clearance <= 0.):
                return False
            if np.all(clearance > distance/(2*intervals)):
                return True
            intervals *= 2
        return False

    def sample_lumen(self, rng: np.random.Generator) -> np.ndarray:
        if self.workspace is not None:
            low, high = self.bounds
            margin = self.bead_radius+self.safety_margin
            return rng.uniform(low+margin, high-margin)
        vessel = self.vessels[int(rng.integers(len(self.vessels)))]
        centre = np.asarray(vessel.start) + rng.random()*(np.asarray(vessel.end)-vessel.start)
        direction = rng.normal(size=3)
        direction /= max(np.linalg.norm(direction), 1e-12)
        radius = max(0., vessel.radius-self.bead_radius-self.safety_margin)
        return centre + direction*radius*rng.random()**(1/3)

    def snapshot(self) -> dict:
        return {
            'kind': self.kind, 'workspace_cm': self.workspace, 'guide_cm': self.guide,
            'units': 'cm', 'start': list(self.start), 'goals': [list(p) for p in self.goals],
            'bead_radius_cm': self.bead_radius, 'safety_margin_cm': self.safety_margin,
            'vessels': [{'start': list(v.start), 'end': list(v.end), 'radius': v.radius}
                        for v in self.vessels],
            'obstacles': [{'center': list(o.center), 'radius': o.radius} for o in self.obstacles],
            'branches': [{'name': b.name, 'points': [list(p) for p in b.points],
                          'radii': list(b.radii)} for b in self.branches],
        }


def _curved_branch(name, controls, start_radius, end_radius, samples_per_span=10):
    """Catmull-Rom centreline through shared junctions, with a gentle taper."""
    points = np.asarray(controls, float)
    padded = np.vstack((points[0], points, points[-1]))
    result = []
    for i in range(len(points)-1):
        p0, p1, p2, p3 = padded[i:i+4]
        for t in np.linspace(0., 1., samples_per_span, endpoint=False):
            result.append(.5*((2*p1)+(-p0+p2)*t
                               +(2*p0-5*p1+4*p2-p3)*t*t
                               +(-p0+3*p1-3*p2+p3)*t*t*t))
    result.append(points[-1])
    radii = np.linspace(start_radius, end_radius, len(result))
    return VesselBranch(name, tuple(tuple(float(v) for v in p) for p in result),
                        tuple(float(v) for v in radii))


def default_scene() -> VascularScene:
    """Winding horizontal vessel with four inclined, sinuous side branches.

    Model Y is horizontal in BOTH camera projections. All geometry uses the
    same XYZ centimetres; branch inclinations are actual height differences.
    """
    def trunk_point(y):
        distance = y+1.15
        return (.12*np.sin(2*np.pi*distance/1.10), y,
                MAIN_HEIGHT_CM+.055*np.sin(2*np.pi*distance/.95))

    root = trunk_point(-1.15)
    forks = tuple(trunk_point(y) for y in (-.65, -.20, .20, .60))
    upward_goal = (-.60, .95, MAIN_HEIGHT_CM+.57)
    downward_goal = (.50, 1.15, MAIN_HEIGHT_CM-.31)
    middle_goal = trunk_point(1.20)
    early_upward_goal = (-.55, .05, MAIN_HEIGHT_CM+.50)
    early_downward_goal = (.50, .35, MAIN_HEIGHT_CM-.35)
    main_nodes = (root, *forks, middle_goal)
    main_branches = tuple(
        _curved_branch('horizontal_trunk_%d' % i,
            [trunk_point(y) for y in np.linspace(start[1], end[1], 5)],
            .20-.005*i, .195-.005*i)
        for i, (start, end) in enumerate(zip(main_nodes[:-1], main_nodes[1:]))
    )
    branches = main_branches + (
        _curved_branch('early_upward_side_branch', [forks[0],
            (-.18,-.43,MAIN_HEIGHT_CM+.18),
            (-.40,-.14,MAIN_HEIGHT_CM+.38), early_upward_goal], .18, .16),
        _curved_branch('early_downward_side_branch', [forks[1],
            (.18,-.01,MAIN_HEIGHT_CM-.14),
            (.36,.20,MAIN_HEIGHT_CM-.27), early_downward_goal], .17, .15),
        _curved_branch('later_upward_side_branch', [forks[2],
            (-.20,.43,MAIN_HEIGHT_CM+.20),
            (-.43,.72,MAIN_HEIGHT_CM+.43), upward_goal], .18, .16),
        _curved_branch('later_downward_side_branch', [forks[3],
            (.17,.78,MAIN_HEIGHT_CM-.12),
            (.36,.99,MAIN_HEIGHT_CM-.24), downward_goal], .17, .15),
    )
    # Add smooth S bends to the side branches. The envelope and its derivative
    # vanish at both ends, keeping all bifurcation and goal positions intact.
    def sinuous_branch(branch):
        points = np.asarray(branch.points).copy()
        t = np.linspace(0., 1., len(points))
        envelope = np.sin(np.pi*t)**2
        points[:, 0] += .11*envelope*np.sin(4*np.pi*t)
        points[:, 2] += .045*envelope*np.sin(4*np.pi*t+np.pi/3)
        return VesselBranch(branch.name, tuple(tuple(p) for p in points), branch.radii)

    branches = tuple(b if b.name.startswith('horizontal_trunk_') else sinuous_branch(b)
                     for b in branches)
    anchor = np.asarray(root, float)

    def world_point(point):
        position = anchor+SCENE_OVERALL_SCALE*(np.asarray(point)-anchor)
        if SCENE_MIRROR_Y:
            position[1] *= -1.
        return tuple(position)

    branches = tuple(VesselBranch(
        branch.name, tuple(world_point(p) for p in branch.points),
        tuple(SCENE_OVERALL_SCALE*SCENE_RADIUS_SCALE*r for r in branch.radii),
    ) for branch in branches)
    vessels = tuple(Vessel(a, b, min(ra, rb)) for branch in branches
                    for a, b, ra, rb in zip(branch.points[:-1], branch.points[1:],
                                             branch.radii[:-1], branch.radii[1:]))
    # Static small wall protrusions along the inlet, trunk and default side
    # route. Offsets lie in each tube's local cross-section, not its tangent.
    def wall_clot(branch_name, fraction, preferred_normal):
        branch = next(b for b in branches if b.name == branch_name)
        points = np.asarray(branch.points)
        i = min(len(points)-2, max(1, round(fraction*(len(points)-1))))
        tangent = points[i+1]-points[i-1]
        tangent /= np.linalg.norm(tangent)
        normal = np.asarray(preferred_normal, float)
        normal -= np.dot(normal, tangent)*tangent
        normal /= np.linalg.norm(normal)
        centre = points[i]+SCENE_OVERALL_SCALE*CLOT_WALL_OFFSET_CM*normal
        return Obstacle(tuple(centre), SCENE_OVERALL_SCALE*CLOT_RADIUS_CM)

    clots = (
        wall_clot('horizontal_trunk_0', .55, (0.,0.,1.)),
        wall_clot('horizontal_trunk_2', .45, (1.,0.,0.)),
        wall_clot('later_upward_side_branch', .55, (0.,0.,1.)),
    )
    return VascularScene(
        vessels=vessels,
        obstacles=clots,
        start=world_point(root),
        goals=tuple(world_point(p) for p in (upward_goal, downward_goal,
                                                  early_upward_goal, early_downward_goal)),
        branches=branches,
    )
