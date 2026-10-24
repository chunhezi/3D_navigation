"""Interactive Qt 3D view of the virtual scene and the physical bead state."""
from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from navigation.reference import get_navigation


class SceneCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 500)
        self.setMouseTracking(True)
        self.yaw, self.pitch, self.zoom = -80., 18., 1.
        self._drag = None
        self.state = {}
        self.show_tree = False
        self.trace = []
        self._last_pair = None
        self._path_id = None
        self._motion_active = False
        self._tube_scene = None
        self._tube_faces = None
        self._tube_normals = None
        self._tube_image_key = None
        self._tube_image = None
        self._scene_kind = None

    def set_state(self, state):
        scene = state.get('navigation_scene')
        if scene is not None and scene.kind != self._scene_kind:
            self._scene_kind = scene.kind
            self.yaw, self.pitch = (-58., 30.) if scene.workspace is not None else (-80., 18.)
        path_id = id(state.get('ref_path_3d'))
        motion = bool(state.get('motion_active'))
        if path_id != self._path_id or (motion and not self._motion_active):
            self.trace.clear()
            self._last_pair = None
        self._path_id, self._motion_active = path_id, motion
        pair = state.get('pair_id')
        if pair != self._last_pair and motion and state.get('draw_trace', False):
            self._last_pair = pair
            p = state.get('position_xyz')
            if p is not None and np.all(state.get('valid_mask', [False]*3)):
                self.trace.append(np.asarray(p, float).copy())
                self.trace = self.trace[-6000:]
        self.state = state
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.position()-self._drag
            self.yaw += delta.x()*.4
            self.pitch = float(np.clip(self.pitch+delta.y()*.4, -75., 75.))
            self._drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def wheelEvent(self, event):
        self.zoom = float(np.clip(self.zoom*1.12**(event.angleDelta().y()/120.), .4, 4.))
        self.update()

    def _project(self, points):
        p = np.asarray(points, float)-self._centre
        yaw, pitch = np.radians([self.yaw, self.pitch])
        horizontal = np.cos(yaw)*p[..., 0]-np.sin(yaw)*p[..., 1]
        depth = np.sin(yaw)*p[..., 0]+np.cos(yaw)*p[..., 1]
        vertical = np.sin(pitch)*depth-np.cos(pitch)*p[..., 2]
        return np.stack([self.width()/2+horizontal*self._scale,
                         self.height()/2+vertical*self._scale], axis=-1)

    def _polyline(self, painter, points, color, width=2, dashed=False):
        pixels = self._project(points)
        if len(pixels) < 2:
            return
        polygon = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in pixels])
        pen = QtGui.QPen(QtGui.QColor(color), width)
        if dashed:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPolyline(polygon)

    def _point(self, painter, position, radius, color):
        xy = self._project(position)
        painter.setPen(QtGui.QPen(QtGui.QColor('#e5edf5'), 1.))
        painter.setBrush(QtGui.QColor(color))
        painter.drawEllipse(QtCore.QPointF(float(xy[0]), float(xy[1])), radius, radius)

    def _draw_curved_tubes(self, painter, scene):
        """Shaded 3D tube surfaces following the collision-model centrelines."""
        if self._tube_scene is not scene:
            faces, normals = [], []
            angles = np.linspace(0., 2*np.pi, 17)[:-1]
            for branch in scene.branches:
                p = np.asarray(branch.points)
                t = np.gradient(p, axis=0)
                t /= np.linalg.norm(t, axis=1)[:, None]
                helper = np.array([0., 0., 1.]) if abs(t[0,2]) < .9 else np.array([1., 0., 0.])
                previous = np.cross(t[0], helper)
                rings, radial = [], []
                for point, tangent, radius in zip(p, t, branch.radii):
                    u = previous-np.dot(previous, tangent)*tangent
                    u /= max(np.linalg.norm(u), 1e-12)
                    v = np.cross(tangent, u)
                    directions = np.cos(angles)[:, None]*u+np.sin(angles)[:, None]*v
                    rings.append(point+radius*directions)
                    radial.append(directions)
                    previous = u
                for i in range(len(rings)-1):
                    for j in range(16):
                        k = (j+1)%16
                        faces.append([rings[i][j], rings[i+1][j], rings[i+1][k], rings[i][k]])
                        normal = radial[i][j]+radial[i][k]+radial[i+1][j]+radial[i+1][k]
                        normals.append(normal/max(np.linalg.norm(normal), 1e-12))
            self._tube_faces, self._tube_normals = np.asarray(faces), np.asarray(normals)
            self._tube_scene = scene
            self._tube_image_key = None
        key = (id(scene), self.yaw, self.pitch, self.zoom, self.width(), self.height())
        if self._tube_image_key != key:
            image = QtGui.QPixmap(self.size())
            image.fill(QtCore.Qt.GlobalColor.transparent)
            brush = QtGui.QPainter(image)
            brush.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            brush.setPen(QtCore.Qt.PenStyle.NoPen)
            faces = self._tube_faces
            centres = faces.mean(axis=1)-self._centre
            yaw, pitch = np.radians([self.yaw, self.pitch])
            depth = (np.sin(yaw)*centres[:,0]+np.cos(yaw)*centres[:,1])*np.cos(pitch)+centres[:,2]*np.sin(pitch)
            pixels = self._project(faces)
            light = np.array([-.35,-.65,1.])
            light /= np.linalg.norm(light)
            brightness = .45+.55*np.maximum(0., self._tube_normals@light)
            for i in np.argsort(depth)[::-1]:
                shade = brightness[i]
                brush.setBrush(QtGui.QColor(int(165*shade+55),int(195*shade+45),int(210*shade+40),95))
                polygon = QtGui.QPolygonF([QtCore.QPointF(float(x),float(y)) for x,y in pixels[i]])
                brush.drawPolygon(polygon)
            brush.end()
            self._tube_image, self._tube_image_key = image, key
        painter.drawPixmap(0, 0, self._tube_image)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#101b2b'))
        scene = self.state.get('navigation_scene')
        plan = self.state.get('navigation_plan')
        if scene is None or plan is None:
            painter.setPen(QtGui.QColor('white'))
            painter.drawText(30, 40, '等待 RRT 规划结果')
            return
        bounds = scene.bounds
        self._centre = bounds.mean(axis=0)
        self._scale = min(self.width()*.86, self.height()*.86)/np.ptp(bounds, axis=0).max()*self.zoom

        if scene.branches:
            self._draw_curved_tubes(painter, scene)
        else:
            for vessel in scene.vessels:
                self._polyline(painter,[vessel.start,vessel.end],'#488ba6',2*vessel.radius*self._scale)
        if scene.guide:
            self._polyline(painter, scene.guide, '#6396ba', 1.2, dashed=True)
        if scene.workspace is not None:
            from itertools import product
            for bits in product((0,1), repeat=3):
                for axis in range(3):
                    if bits[axis] == 0:
                        end = list(bits)
                        end[axis] = 1
                        self._polyline(painter, [bounds[np.array(bits), np.arange(3)],
                                               bounds[np.array(end), np.arange(3)]], '#314960', 1.)
        if self.show_tree:
            for i in range(1, len(plan.tree_nodes)):
                if plan.tree_parents[i] < 0:
                    continue
                self._polyline(painter, [plan.tree_nodes[plan.tree_parents[i]], plan.tree_nodes[i]], '#356274', .7)
        # Orthographic sphere silhouettes with shading. Overlapping spheres
        # form each irregular solid, matching the union used for collision.
        for obstacle in sorted(scene.obstacles, key=lambda o:o.center[0], reverse=True):
            xy = self._project(obstacle.center)
            radius = obstacle.radius*self._scale
            centre = QtCore.QPointF(float(xy[0]), float(xy[1]))
            gradient = QtGui.QRadialGradient(centre, radius,
                QtCore.QPointF(float(xy[0]-.3*radius), float(xy[1]-.35*radius)))
            gradient.setColorAt(0., QtGui.QColor('#ffa6a0'))
            gradient.setColorAt(.55, QtGui.QColor('#d94d66'))
            gradient.setColorAt(1., QtGui.QColor('#762b48'))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(gradient))
            painter.drawEllipse(centre, radius, radius)
        self._polyline(painter, plan.path, '#f2cd68', 2.5)
        if len(self.trace) > 1:
            self._polyline(painter, self.trace, '#62e1b2', 1.6)
        self._point(painter, scene.start, 5., '#66c9ee')
        for goal in scene.goals:
            self._point(painter, goal, 4., '#a198b9')
        self._point(painter, plan.path[-1], 6., '#e9b0ee')
        target = self.state.get('target_xyz')
        if target is not None:
            self._point(painter, target, 5., '#ff9d48')
        position = self.state.get('position_xyz')
        if position is not None and np.all(np.isfinite(position)):
            self._point(painter, position, max(4., scene.bead_radius*self._scale), '#62e1b2')
        origin = np.array([bounds[0,0], bounds[0,1], bounds[0,2]])
        for i, color in enumerate(('#fa8585', '#84dca6', '#8cb6ff')):
            end = origin.copy()
            end[i] += .3
            self._polyline(painter, [origin, end], color, 2.)
            xy = self._project(end)
            painter.setPen(QtGui.QColor(color))
            painter.drawText(QtCore.QPointF(float(xy[0]+5), float(xy[1])), 'XYZ'[i]+' (cm)')
        painter.setPen(QtGui.QColor('#a8b8cd'))
        painter.drawText(18, self.height()-18, '拖动旋转 · 滚轮缩放 · 所有轴使用相同厘米比例')


class NavigationWindow(QtWidgets.QDialog):
    def __init__(self, *, get_state, replan, can_replan, parent=None):
        super().__init__(parent)
        self.setFont(QtGui.QFont('Microsoft YaHei', 10))
        self.setWindowTitle('虚拟分叉血管 · 三维 RRT 导航')
        self.resize(940, 790)
        self.get_state, self.replan, self.can_replan = get_state, replan, can_replan
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel('实体磁珠导航 / 虚拟血管与障碍物')
        self.title = title
        title.setStyleSheet('font-size:18px; font-weight:600; padding:6px')
        layout.addWidget(title)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel('终点分支'))
        self.goals = QtWidgets.QComboBox()
        self.goals.addItems(['后段向上侧枝末端（默认）', '后段向下侧枝末端',
                            '前段向上侧枝末端', '前段向下侧枝末端'])
        from navigation.reference import GOAL_BRANCH
        self.goals.setCurrentIndex(GOAL_BRANCH)
        controls.addWidget(self.goals)
        self.plan_button = QtWidgets.QPushButton('RRT 重新规划')
        self.plan_button.clicked.connect(self._replan)
        controls.addWidget(self.plan_button)
        tree = QtWidgets.QCheckBox('显示搜索树')
        controls.addWidget(tree)
        controls.addStretch()
        layout.addLayout(controls)
        self.canvas = SceneCanvas()
        tree.toggled.connect(self._tree_changed)
        layout.addWidget(self.canvas, 1)
        legend = QtWidgets.QLabel('半透明管道：虚拟血管    红色：静态障碍物    蓝色虚线：引导曲线    黄色：RRT 路径    橙色：当前目标    绿色：磁珠/轨迹')
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(QtWidgets.QLabel('先完成双相机标定，再在主窗口点击 start motion。入场尾巴隐藏，达到起点后开始导航。'))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(150)
        self.refresh()

    def _tree_changed(self, checked):
        self.canvas.show_tree = checked
        self.canvas.update()

    def _replan(self):
        if not self.can_replan():
            return
        try:
            self.replan(self.goals.currentIndex())
            self.refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, '规划失败', str(exc))

    def refresh(self):
        state = self.get_state()
        scene = state.get('navigation_scene')
        if scene is not None:
            static = scene.workspace is not None
            slalom = scene.kind == 'slalom'
            names = (['三维阶梯路径终点'] if slalom else ['正余弦轨迹终点'] if static else
                     ['后段向上侧枝末端（默认）', '后段向下侧枝末端',
                      '前段向上侧枝末端', '前段向下侧枝末端'])
            if names != [self.goals.itemText(i) for i in range(self.goals.count())]:
                from navigation.reference import GOAL_BRANCH
                self.goals.clear()
                self.goals.addItems(names)
                self.goals.setCurrentIndex(0 if static else GOAL_BRANCH)
            self.title.setText('正余弦三维轨迹 / 不规则静态障碍物 / RRT 绕行' if static
                               else '实体磁珠导航 / 虚拟血管与障碍物')
            self.setWindowTitle('三维静态避障' if static else '虚拟分叉血管 · 三维 RRT 导航')
            if slalom:
                self.title.setText('三维阶梯折线 / 三组静态障碍物 / RRT 绕行')
                self.setWindowTitle('三维阶梯避障')
        self.canvas.set_state(state)
        self.plan_button.setEnabled(self.can_replan())
        self.goals.setEnabled(self.can_replan())
        plan = state.get('navigation_plan')
        if plan is None:
            self.status.setText('等待切换到 navigate_3d_mode')
            return
        info = state.get('reference_info', {})
        phase = info.get('phase', 'ready')
        labels = {'entry': '缓慢入场', 'ready': '规划就绪', 'tracking': '路径跟随',
                  'waiting_for_bead': '等待磁珠跟上', 'goal_reached': '已到达终点',
                  'outside_virtual_free_space': '磁珠超出虚拟可行区域，目标暂停推进',
                  'correction_segment_blocked': '连接段受阻，目标暂停推进'}
        status = labels.get(info.get('navigation_status', phase), phase)
        if phase == 'entry':
            status = '缓慢入场'
        self.status.setText(
            f'RRT：{len(plan.tree_nodes)} 个节点 · {plan.elapsed_s:.3f} 秒 · 路径 {plan.length_cm:.2f} cm'
            f'\n状态：{status} · 进度 {100*info.get("progress",0.):.1f}%'
        )


def preview():
    """Standalone planning preview, deliberately without any device imports."""
    from navigation.reference import get_navigation_reference, reset_navigation
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import sys
    if '--static' in sys.argv:
        from navigation import reference
        reference.SCENE_KIND = 'static_obstacles'
    if '--slalom' in sys.argv:
        from navigation import reference
        reference.SCENE_KIND = 'slalom'
    reset_navigation()
    def state():
        scene, plan = get_navigation()
        return dict(navigation_scene=scene, navigation_plan=plan,
                    ref_path_3d=plan.path, target_xyz=plan.path[0])
    def replan(goal):
        from navigation import reference
        reference.GOAL_BRANCH = goal
        reset_navigation()
    window = NavigationWindow(get_state=state, replan=replan, can_replan=lambda:True)
    window.setWindowTitle('虚拟场景规划预览（未连接硬件）')
    window.show()
    app.exec()
