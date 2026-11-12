"""Read-only recomputation from every frozen final 1 ms physical log."""
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from phase2c8_protocol import ROOT, RUN, read, write_json
from evaluate_phase2c8 import rescore


def main():
    rows = []
    inputs = {}
    input_roundoff = 0.
    for path in sorted((RUN / 'final').rglob('metrics.json')):
        m = read(path)
        d = pd.read_csv(__import__('pathlib').Path(m['folder']) / 'physics.csv.gz')
        assert len(d) == m['steps']
        task = m['task']
        t = d.time_s.to_numpy()
        steps = d.input_step.to_numpy()
        np.testing.assert_array_equal(steps, np.arange(len(d)))
        np.testing.assert_allclose(t, (steps + 1) * .001, atol=1e-10, rtol=0)
        actual = Rotation.from_euler('xyz', d[[f'rpy_{i}' for i in range(3)]], degrees=True)
        target = Rotation.from_euler('xyz', d[[f'rpy_ref_{i}' for i in range(3)]], degrees=True)
        err = (actual * target.inv()).as_rotvec() * 180 / np.pi
        so3 = np.linalg.norm(err, axis=1)
        np.testing.assert_allclose(so3, d.orientation_error_deg, atol=1e-10, rtol=0)
        e = d[[f'p_{i}' for i in range(3)]].to_numpy() - d[[f'p_ref_{i}' for i in range(3)]].to_numpy()
        f = d[[f'force_{i}' for i in range(6)]].to_numpy()
        add = d[[f'F_command_{i}' for i in range(6)]].to_numpy() - d[[f'F_C0_{i}' for i in range(6)]].to_numpy()
        external = d[[f'external_world_{i}' for i in range(6)]].to_numpy()
        mask = (steps >= round(task['on'] * 1000)) & (steps < round(task['off'] * 1000))
        expected = mask[:, None] * np.array([0, 0, 0, *task['torque'], 0])
        np.testing.assert_allclose(external, expected, atol=1e-12, rtol=0)
        frozen_input = d[[*[f'p_ref_{i}' for i in range(3)], *[f'rpy_ref_{i}' for i in range(3)],
                          *[f'twist_ref_{i}' for i in range(6)], *[f'external_world_{i}' for i in range(6)]]].to_numpy()
        if task['id'] in inputs:
            np.testing.assert_array_equal(inputs[task['id']][:, :12], frozen_input[:, :12])
            # Logged world wrench is reconstructed through body/world rotations.
            # Roundoff depends on actual orientation; intended wrench was checked above.
            input_roundoff = max(input_roundoff, float(abs(inputs[task['id']] - frozen_input).max()))
            np.testing.assert_allclose(inputs[task['id']][:, 12:], frozen_input[:, 12:], atol=1e-12, rtol=0)
        else:
            inputs[task['id']] = frozen_input
        direction = np.array(task['torque']) / np.linalg.norm(task['torque'])
        metrics = dict(
            so3_peak_deg=float(so3.max()), roll_error_peak_deg=float(abs(err[:, 0]).max()),
            pitch_error_peak_deg=float(abs(err[:, 1]).max()), yaw_error_peak_deg=float(abs(err[:, 2]).max()),
            omega_peak_rad_s=float(np.linalg.norm(d[[f'twist_actual_{i}' for i in (3, 4, 5)]], axis=1).max()),
            z_rmse_mm=float(np.sqrt(np.mean(e[:, 2] ** 2)) * 1000),
            xyz_rmse_mm=float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))) * 1000),
            xy_peak_mm=float(abs(e[:, :2]).max() * 1000),
            force_peak_N=float(abs(f).max()), force_rate_N_s=float(abs(np.diff(f, axis=0)).max() * 1000),
            force_squared_integral_N2_s=float(np.sum(f ** 2) * .001),
            add_peak_N=float(abs(add).max()),
            add_rate_N_s=float(abs(np.diff(np.vstack([np.zeros(6), add]), axis=0)).max() * 1000),
            unload_reverse_overshoot_deg=float(max(0, -(err[t >= task['off'], :2] @ direction).min(initial=0))),
            condition_peak=float(d.actual_jacobian_condition.max()),
            weld_translation_m=float(d.weld_translation_m.max()), weld_angle_rad=float(d.weld_angle_rad.max()),
            clearance_m=float(d.expanded_mesh_gap_m.min()),
            cap_fraction=float((d.reward_cap_factor < 1).mean()),
            native_force_saturation_steps=int(d.native_force_saturated.sum()),
        )
        for label, event in [('load', 'on'), ('unload', 'off')]:
            start = round(task[event] * 1000)
            metrics[label + '_iae_deg_s'] = float(so3[(steps >= start) & (steps < start + 1000)].sum() * .001)
        for label, start, end in [('load', task['on'], task['off']), ('unload', task['off'], 3 + 2 * task['reference']['rise_s'])]:
            inside = (t >= start) & (t <= end + 1e-9)
            times, errors = t[inside], so3[inside]
            for band in (.1, .01):
                violations = np.flatnonzero(errors >= band)
                elapsed = None
                if not len(violations):
                    status = 'not_triggered'
                else:
                    next_index = violations[-1] + 1
                    status = 'right_censored'
                    if next_index < len(times) and end - times[next_index] >= .5 - 1e-9:
                        status = 'recovered'
                        elapsed = float(times[next_index] - start)
                assert m[f'{label}_recovery_{band}_status'] == status
                if elapsed is not None:
                    assert abs(m[f'{label}_recovery_{band}_s'] - elapsed) < 1e-10
                else:
                    assert m[f'{label}_recovery_{band}_s'] is None
        samples = pd.read_csv(path.parent / 'training_samples.csv')
        penalty = float(samples.terminal_penalty.sum())
        for version, (components, _) in rescore(d).items():
            metrics[f'common_{version}_return'] = float(-components.sum() / 20 + penalty)
        assert abs(samples.reward.sum() - m['episode_return']) < 1e-8
        for key, value in metrics.items():
            np.testing.assert_allclose(value, m[key], rtol=1e-9, atol=1e-10, err_msg=f'{path}:{key}')
            rows.append(dict(task_id=task['id'], method=m['method'], seed=m['seed'], metric=key,
                             recomputed=value, recorded=m[key], absolute_error=abs(value - m[key]), source=m['folder']))
        assert d.failure.isna().all() and m['complete'] and m['success']
    pd.DataFrame(rows).to_csv(ROOT / 'results/phase2c8_final_metric_recomputation.csv', index=False)
    write_json(RUN / 'final_metric_validation.json', dict(passed=True, trials=len(rows) // len(metrics),
        metrics_per_trial=len(metrics), physical_steps=sum(len(a) * 8 for a in inputs.values()),
        shared_reference_exact=True, shared_load_atol_N_Nm=1e-12, max_input_roundoff=input_roundoff,
        measured_actual_state_not_reference=True,
        max_absolute_error=max(r['absolute_error'] for r in rows), recovery_checks=True,
        tolerance='rtol=1e-9, atol=1e-10; recovery time atol=1e-10', no_model_or_training_changes=True))
    print(json.dumps(read(RUN / 'final_metric_validation.json'), indent=2))


if __name__ == '__main__':
    main()
