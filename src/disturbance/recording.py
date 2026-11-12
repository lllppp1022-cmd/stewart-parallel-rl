"""Colour only MjvScene geoms, never the physics model; capture actual live MjData."""
import types,hashlib,json
import mujoco,numpy as np,imageio.v2 as imageio
from ..rendering import camera
from ..visualization.enhanced.style import leg_mapping,rgba

def install_colored_capture(recorder,model):
    mapping={gid:rgba(row['leg']-1) for row in leg_mapping(model) for gid in row['geoms']}
    def capture(self,m,d,snapshot,state_index):
        if not self.enabled:return
        try:
            if self.renderer is None:
                m.vis.global_.offwidth=1216;m.vis.global_.offheight=912;self.renderer=mujoco.Renderer(m,width=1216,height=912);self.writer=imageio.get_writer(self.folder/'actual_rgb.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16)
            self.renderer.update_scene(d,camera=camera())
            for geom in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
                if geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and geom.objid in mapping:geom.rgba[:]=mapping[geom.objid];geom.matid=-1
            self.writer.append_data(self.renderer.render().copy());item=dict(frame=self.nframes,state_index=state_index,step=snapshot.step,time_s=snapshot.time_s,qpos_sha256=hashlib.sha256(d.qpos.tobytes()).hexdigest(),source='ACTUAL LIVE EXECUTION RGB',source_mode='DYNAMIC DISTURBANCE RESPONSE EXPERIMENT',external_world=snapshot.data['external_world'])
            self.frames.write(json.dumps(item)+'\n');self.frames.flush();self.nframes+=1
        except Exception as exc:
            self.render_failures.append(dict(time_s=float(d.time),error=repr(exc)));self.enabled=False;self.event(dict(type='recording_error',reason=repr(exc)))
    recorder.capture=types.MethodType(capture,recorder)
