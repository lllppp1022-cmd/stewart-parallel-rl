"""Correct only the color legend of our derived videos; archive prior render."""
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from phase2c7v_common import ROOT,RUN,read,write_json,sha
def main():
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for kind in ('training','comparison'):
        proof=RUN/f'{kind}_video_validation.json';r=read(proof)
        from pathlib import Path
        target=Path(r['path']);archive=RUN/'draft_videos'/target.name
        target.resolve().relative_to(ROOT.resolve());archive.resolve().relative_to(RUN.resolve())
        if archive.exists():raise FileExistsError(archive)
        assert sha(target)==r['sha'];archive.parent.mkdir(exist_ok=True);target.rename(archive)
        n=0
        with imageio.get_reader(archive) as reader,imageio.get_writer(target,fps=25,codec='libx264',quality=9,macro_block_size=8) as writer:
            for frame in reader:
                image=Image.fromarray(frame);draw=ImageDraw.Draw(image);draw.rectangle((978,1048,1919,1079),fill='#0c141f')
                draw.text((978,1050),'Blue/orange/green/red/purple/cyan: listed order; Nm and N.',font=font,fill='#9cabb9');writer.append_data(np.asarray(image));n+=1
        with imageio.get_reader(target) as reader:assert sum(1 for _ in reader)==n==r['frames']
        r.update(sha=sha(target),legend_corrected=True,previous_render=str(archive));write_json(proof,r);print(kind,n,flush=True)
if __name__=='__main__':main()
