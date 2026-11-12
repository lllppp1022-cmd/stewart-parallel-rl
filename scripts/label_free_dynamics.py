"""Clarify only the initial zero-force caption; preserve original live capture."""
import _bootstrap
from src.model import ROOT,sha,write_json
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
source=ROOT/'assets/videos/phase0_simulation.mp4';original=ROOT/'assets/videos/phase0_simulation_original_capture.mp4';target=ROOT/'assets/videos/phase0_captioned.mp4'
if original.exists():raise FileExistsError('Caption edit already archived')
before=sha(source);reader=imageio.get_reader(source);writer=imageio.get_writer(target,fps=10,codec='libx264',quality=8,macro_block_size=16)
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
try:
 for index,frame in enumerate(reader):
  if index<50:
   im=Image.fromarray(frame);draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,26),fill=(12,23,35));draw.text((12,5),f'FREE DYNAMICS / ZERO CTRL | zero | t={(index+1)/10:.3f} s',font=font,fill=(230,242,250));frame=np.asarray(im)
  writer.append_data(frame)
finally:reader.close();writer.close()
source.rename(original);target.rename(source)
write_json(ROOT/'results/phase0_caption_provenance.json',dict(original=str(original.relative_to(ROOT)),original_sha256=before,published=str(source.relative_to(ROOT)),published_sha256=sha(source),only_semantic_edit='First 50 frames: zero-force baseline explicitly labelled Free Dynamics. Body imagery is the existing live capture; no environment or policy called.',frames=450,reencoding=True,physics_logs_unchanged=True))
