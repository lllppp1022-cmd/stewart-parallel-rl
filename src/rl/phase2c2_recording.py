"""Actual live RGB, fixed camera, enlarged viewport not enlarged physical motion."""
import types,hashlib,json
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from ..visualization.enhanced.style import leg_mapping,rgba

def install_capture(owner):
    recorder=owner.recorder;mapping={gid:rgba(row['leg']-1) for row in leg_mapping(owner.model) for gid in row['geoms']}
    def capture(self,m,d,snapshot,state_index):
        if not self.enabled:return
        try:
            if self.renderer is None:
                m.vis.global_.offwidth=1280;m.vis.global_.offheight=720;self.renderer=mujoco.Renderer(m,width=1280,height=720);self.writer=imageio.get_writer(self.folder/'actual_rgb.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16)
            cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.70];cam.distance=2.2;cam.azimuth=135;cam.elevation=-17
            self.renderer.update_scene(d,camera=cam)
            for g in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
                if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid in mapping:g.rgba[:]=mapping[g.objid];g.matid=-1
            rgb=self.renderer.render().copy();img=Image.fromarray(rgb);draw=ImageDraw.Draw(img);ctx=owner.context;z=(d.site_xpos[owner.site_id,2]-owner.geometry.home_position[2])*1000;zr=(owner.ref.p[2]-owner.geometry.home_position[2])*1000
            draw.rectangle((0,0,1280,95),fill=(10,18,30));label='ACTUAL STOCHASTIC TRAINING' if ctx.get('mode')=='Actual Training' else ctx.get('mode','Baseline Comparison')
            lines=[f'{label} | REAL 1 kHz MuJoCo integration | no displacement magnification',f't={d.time:.3f}s  episode={owner.episode} decision={ctx.get("global_decision",-1)} policy={ctx.get("policy_version",-1)}  action={owner.residual.held_action.normalized:+.3f}',f'Z actual={z:.4f} mm  target={zr:.4f} mm  error={(z-zr)*1000:+.1f} um  proxy gap={snapshot.data["proxy_gap"]*1000:.2f} mm',f'reward cost={sum(owner.last_cost.values()):.4f}  status={owner.failure or "RUNNING"} | SIDE GAUGE: measured Z, 120 pixels/mm (schematic)']
            for k,s in enumerate(lines):draw.text((12,7+21*k),s,fill='white')
            draw.rectangle((1130,145,1270,670),fill=(10,18,30))
            for mm in range(5):
                y=625-mm*120;draw.line((1140,y,1250,y),fill=(70,80,95));draw.text((1140,y-18),f'{mm} mm',fill='white')
            for x,val,col in [(1185,z,(30,220,140)),(1225,zr,(255,190,40))]:draw.rectangle((x-8,625-120*val-3,x+8,625-120*val+3),fill=col)
            self.writer.append_data(np.array(img));self.frames.write(json.dumps(dict(frame=self.nframes,state_index=state_index,time_s=float(d.time),qpos_sha256=hashlib.sha256(d.qpos.tobytes()).hexdigest(),source='ACTUAL LIVE RGB',policy_version=ctx.get('policy_version',-1),global_decision=ctx.get('global_decision',-1),z_mm=z,reference_z_mm=zr))+'\n');self.frames.flush();self.nframes+=1
        except Exception as exc:
            self.render_failures.append(dict(time_s=float(d.time),error=repr(exc)));self.enabled=False;self.event(dict(type='recording_error',reason=repr(exc)))
    recorder.capture=types.MethodType(capture,recorder)
