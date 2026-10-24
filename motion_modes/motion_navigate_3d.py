"""Physical-bead navigation through a virtual vascular scene using 3D RRT."""
from motion_modes.motion_stabilize_3d import Stabilize3DMotionMode
from navigation.reference import get_navigation


class Navigate3DMotionMode(Stabilize3DMotionMode):
    name = 'navigate_3d'
    reference_mode = 'rrt_navigation'
    reference_scene = 'vessels'

    def reset(self):
        from navigation import reference
        reference.SCENE_KIND = self.reference_scene
        super().reset()

    def get_overlay_state(self):
        state = super().get_overlay_state()
        scene, plan = get_navigation()
        state['navigation_scene'] = scene
        state['navigation_plan'] = plan
        return state


class StaticObstacle3DMotionMode(Navigate3DMotionMode):
    name = 'static_obstacle_3d'
    reference_scene = 'static_obstacles'


class Slalom3DMotionMode(Navigate3DMotionMode):
    name = 'slalom_3d'
    reference_scene = 'slalom'
