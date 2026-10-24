"""Project the SAME virtual vessel/obstacle geometry into either camera."""
import cv2
import numpy as np


def draw_navigation_scene(overlay, mask, scene, project):
    def line(points, color, closed=False):
        pixels = np.asarray(project(np.asarray(points, float)), np.int32)
        cv2.polylines(overlay, [pixels], closed, color, 1, cv2.LINE_AA)
        cv2.polylines(mask, [pixels], closed, 255, 1, cv2.LINE_AA)

    angles = np.linspace(0., 2*np.pi, 33)
    if scene.guide:
        line(scene.guide, (125, 105, 85))
    if scene.workspace is not None:
        from itertools import product
        bounds = np.asarray(scene.workspace)
        for bits in product((0, 1), repeat=3):
            for axis in range(3):
                if bits[axis] == 0:
                    end = list(bits)
                    end[axis] = 1
                    line([bounds[np.array(bits), np.arange(3)],
                          bounds[np.array(end), np.arange(3)]], (90, 75, 55))
    for branch in scene.branches:
        centres = np.asarray(branch.points, float)
        tangents = np.gradient(centres, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        helper = np.array([0., 0., 1.]) if abs(tangents[0, 2]) < .9 else np.array([1., 0., 0.])
        previous = np.cross(tangents[0], helper)
        rings = []
        for centre, tangent, radius in zip(centres, tangents, branch.radii):
            u = previous-np.dot(previous, tangent)*tangent
            u /= max(np.linalg.norm(u), 1e-12)
            v = np.cross(tangent, u)
            rings.append(centre+radius*(np.cos(angles)[:, None]*u+np.sin(angles)[:, None]*v))
            previous = u
        rings = np.asarray(rings)
        for i in sorted(set(range(0, len(rings), 12)) | {len(rings)-1}):
            line(rings[i], (170, 125, 40), True)
        for i in (0, 8, 16, 24):
            line(rings[:, i], (170, 125, 40))
    for vessel in (() if scene.branches else scene.vessels):
        a, b = np.asarray(vessel.start), np.asarray(vessel.end)
        direction = (b-a)/np.linalg.norm(b-a)
        helper = np.array([0., 0., 1.]) if abs(direction[2]) < .9 else np.array([1., 0., 0.])
        u = np.cross(direction, helper)
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        offsets = vessel.radius*(np.cos(angles)[:, None]*u + np.sin(angles)[:, None]*v)
        for centre in (a, (a+b)/2, b):
            line(centre+offsets, (170, 125, 40), True)
        for i in (0, 8, 16, 24):
            line([a+offsets[i], b+offsets[i]], (170, 125, 40))
    for obstacle in scene.obstacles:
        for axes in ((0, 1), (0, 2), (1, 2)):
            points = np.tile(obstacle.center, (len(angles), 1)).astype(float)
            points[:, axes[0]] += obstacle.radius*np.cos(angles)
            points[:, axes[1]] += obstacle.radius*np.sin(angles)
            line(points, (20, 40, 230), True)
