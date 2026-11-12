"""Project true logged geometry onto original RGB. No integration or motion scaling."""
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from ...phase1.plant import create_plant
from ...rendering import camera
from .style import LEG_COLORS,leg_mapping
from .mirror import anchors

def project_points(scene,points,width,height):
    a,b=scene.camera;pos=(a.pos.astype(float)+b.pos)/2;forward=(a.forward.astype(float)+b.forward)/2;forward/=np.linalg.norm(forward);up=(a.up.astype(float)+b.up)/2;up/=np.linalg.norm(up);right=np.cross(forward,up);right/=np.linalg.norm(right)
    near=float(a.frustum_near);top=float(a.frustum_top);bottom=float(a.frustum_bottom);half_y=(top-bottom)/2;half_x=half_y*width/height;center_y=(top+bottom)/2;center_x=float(a.frustum_center)
    result=[]
    for p in points:
        v=np.asarray(p)-pos;depth=float(v@forward)
        if depth<=near:result.append(None);continue
        result.append((width*(.5+((v@right)*near/depth-center_x)/(2*half_x)),height*(.5-((v@up)*near/depth-center_y)/(2*half_y))))
    return result

class ProjectionOverlay:
    def __init__(self,condition='nominal'):
        self.model=create_plant(condition);self.data=mujoco.MjData(self.model);self.mapping=leg_mapping(self.model);self.scene=mujoco.MjvScene(self.model,maxgeom=1000)
        try:self.font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',22)
        except OSError:self.font=ImageFont.load_default()
    def apply(self,rgb,qpos,qvel=None,ctrl=None,time_s=0):
        d=self.data;d.qpos[:]=qpos
        if qvel is not None:d.qvel[:]=qvel
        if ctrl is not None:d.ctrl[:]=ctrl
        d.time=time_s;mujoco.mj_forward(self.model,d);mujoco.mjv_updateScene(self.model,d,mujoco.MjvOption(),None,camera(),mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        h,w=rgb.shape[:2];pixels=project_points(self.scene,anchors(self.model,d,self.mapping),w,h);im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);order=sorted(range(6),key=lambda i:pixels[i][0] if pixels[i] else 0)
        for rank,i in enumerate(order):
            point=pixels[i]
            if point is None:continue
            x,y=point;left=rank<3;tx=22 if left else w-137;ty=180+(rank%3)*130;color=LEG_COLORS[i];draw.line((x,y,tx+(108 if left else 0),ty+17),fill=color,width=2);draw.ellipse((x-5,y-5,x+5,y+5),fill=color);draw.rounded_rectangle((tx,ty,tx+112,ty+34),radius=5,fill='#102c3c',outline=color,width=2);draw.text((tx+8,ty+3),f'Leg {i+1}',font=self.font,fill=color)
        return np.asarray(im),pixels
