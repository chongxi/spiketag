from vispy.scene import BaseCamera
from vispy.geometry import Rect

class YSyncCamera(BaseCamera):
    '''
        Camera only be sync the Y coordinate, not sync with X coordinate.
    '''
    def set_state(self, state=None, **kwargs):
        D = state or {}
        if 'rect' not in D:
            return
        for cam in self._linked_cameras:
            r = Rect(D['rect'])
            if cam is self._linked_cameras_no_update:
                continue
            try:
                cam._linked_cameras_no_update = self
                cam_rect = cam.get_state()['rect']
                r.left = cam_rect.left
                r.right = cam_rect.right
                cam.set_state({'rect':r})
            finally:
                cam._linked_cameras_no_update = None
