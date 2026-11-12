"""Annotate actual RGB from the perturbed-plant baseline trial."""
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from audit_phase2c9_dynamics import ROOT, RUN
from phase2c8_protocol import read, sha, write_json


def main():
    folder = RUN / "video_capture/09/episode_0000"
    raw = folder / "actual_rgb.mp4"
    target = ROOT / "assets/videos/stewart_phase2c9_unknown_dynamics.mp4"
    if target.exists():
        raise FileExistsError(target)
    physical = pd.read_csv(folder / "physics.csv.gz")
    states = np.load(folder / "integration_states.npz")["states"]
    frames = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
    assert not read(folder / "summary.json")["recording_errors"]
    original = np.load(RUN / "preflight_v2/09/episode_0000/integration_states.npz")["states"]
    np.testing.assert_array_equal(original, states)
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 20)
    change, previous = 0., None
    with imageio.get_reader(raw) as source, imageio.get_writer(target, fps=25, macro_block_size=16) as output:
        for j, rgb in enumerate(source):
            index = frames[j]["state_index"]
            r = physical.iloc[index]
            assert abs(r.time_s - frames[j]["time_s"]) < 1e-9
            assert hashlib.sha256(states[index, 1:44].tobytes()).hexdigest() == frames[j]["qpos_sha256"]
            if previous is not None:
                change = max(change, float(np.abs(rgb.astype(float) - previous).mean()))
            previous = rgb.astype(float)
            canvas = Image.new("RGB", (1280, 480), "#101a25")
            canvas.paste(Image.fromarray(rgb).resize((640, 480)), (0, 0))
            draw = ImageDraw.Draw(canvas)
            lines = [
                "ACTUAL DYNAMICS | FROZEN BASELINE",
                "Plant mass 22 kg | nominal model 20 kg",
                f"Time {r.time_s:.3f} s | combined 50 mm motion",
                f"Z ref / actual: {r.reference_z_from_home_m*1000:.3f} / {r.actual_z_from_home_m*1000:.3f} mm",
                f"Roll ref / actual: {r.rpy_ref_0:.4f} / {r.rpy_0:.4f} deg",
                f"Pitch ref / actual: {r.rpy_ref_1:.4f} / {r.rpy_1:.4f} deg",
                f"Mx / My: {r.external_world_3:+.3f} / {r.external_world_4:+.3f} Nm",
                f"SO3 error: {r.orientation_error_deg:.6f} deg",
                "RL requested / accepted: 0 / 0 Nm",
                "Leg forces [N]: " + " ".join(f"{r[f'force_{k}']:.1f}" for k in range(6)),
            ]
            for k, line in enumerate(lines):
                draw.text((655, 24 + k * 37), line, fill="white", font=font)
            output.append_data(np.asarray(canvas))
    assert j + 1 == len(frames) and change > 0
    with imageio.get_reader(target) as result:
        assert result.count_frames() == len(frames)
    write_json(RUN / "video_validation.json", dict(path=str(target), sha256=sha(target),
        frames=len(frames), fps=25, seconds=len(frames)/25, pixel_change=change,
        source=str(raw), source_sha256=sha(raw), bitwise_preflight_trajectory=True,
        frame_timestamp_and_qpos_sha_validated=True, actual_training=False))


if __name__ == "__main__":
    main()
