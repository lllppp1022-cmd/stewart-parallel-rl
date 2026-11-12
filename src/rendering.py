import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def camera():
    cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.55];cam.distance=2.65;cam.azimuth=135;cam.elevation=-19;return cam
def annotated_frame(renderer,model,data,row,label):
    renderer.update_scene(data,camera=camera());frame=renderer.render().copy();im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,960,122),fill=(12,23,35))
    try:font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
    except OSError:font=ImageFont.load_default()
    mode='FREE DYNAMICS / ZERO CTRL' if label=='zero' else 'DYNAMIC ACTUATOR SIMULATION'
    text=[f'{mode} | {label} | t={data.time:.3f} s',f'ctrl = force (N), six leg channels; all six platform disturbance channels = 0',f'position error {row["position_error_m"]*1000:.4f} mm | angle error {np.rad2deg(row["orientation_error_rad"]):.5f} deg',f'max weld gap {row["max_site_gap_m"]*1e6:.3f} um | peak current force {max(abs(data.actuator_force[:6])):.2f} N',f'Live MjData + mj_step | not kinematic playback | no collision-safety claim']
    for j,line in enumerate(text):draw.text((12,5+j*23),line,font=font,fill=(230,242,250))
    return np.asarray(im)
