"""Static obstacles along a spatial staircase with straight XYZ legs."""
import numpy as np
from navigation.scene import VascularScene, Obstacle


# Horizontal/transverse legs alternate with vertical rises: a staircase in
# side YZ and angular left/right turns in top XY. Coordinates are cm.
STAIR_WAYPOINTS_CM = np.array([
    [0.,1.,.25], [0.,.65,.25], [.40,.65,.25], [.40,.65,.45],
    [.40,.05,.45], [-.35,.05,.45], [-.35,.05,.65],
    [-.35,-.60,.65], [.10,-.60,.65], [.10,-.60,.85], [.10,-1.05,.85],
], dtype=float)


def curve(t):
    t = np.clip(np.asarray(t, float), 0., 1.)
    scaled = t*(len(STAIR_WAYPOINTS_CM)-1)
    index = np.minimum(scaled.astype(int), len(STAIR_WAYPOINTS_CM)-2)
    u = (scaled-index)[..., None]
    return (1-u)*STAIR_WAYPOINTS_CM[index]+u*STAIR_WAYPOINTS_CM[index+1]


def slalom_scene():
    # Sixteen samples per leg: the RRT anchor stride (8) includes every corner.
    guide = curve(np.linspace(0., 1., 16*(len(STAIR_WAYPOINTS_CM)-1)+1))
    obstacles = []
    # Three lobed pillars with alternating high/low centres and side offsets.
    # All pillars are stationary; the reference must weave in three dimensions.
    # Put obstacles beside the long traverses rather than on the riser
    # corners, so obstacle bypasses do not erase the staircase structure.
    for t, offset in ((.04, (.17,0.,-.025)), (.35, (.04,0.,.025)),
                      (.65, (-.035,0.,.025))):
        centre = curve(t)+offset
        for delta, radius in (((0.,0.,0.),.12), ((.035,.015,.10),.10),
                              ((-.035,-.015,-.09),.095)):
            obstacles.append(Obstacle(tuple(centre+delta), radius))
    return VascularScene(
        vessels=(), obstacles=tuple(obstacles), start=tuple(guide[0]),
        goals=(tuple(guide[-1]),), workspace=((-0.85,-1.25,.10),(.85,1.25,1.05)),
        guide=tuple(tuple(p) for p in guide), kind='slalom',
    )
