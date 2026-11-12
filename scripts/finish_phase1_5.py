"""Recompute delivery evidence from immutable execution journals, not claimed metrics."""
import _bootstrap
from pathlib import Path
import json,hashlib,shutil,xml.etree.ElementTree as ET
import numpy as np,pandas as pd,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,write_json

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    backend=read(ROOT/'results/phase1_5_backend_validation.json');base=ROOT/backend['run_directory']/'observer_off';states=np.load(base/'integration_states.npz')['states']
    proof=ROOT/'results/runs/phase1_5_session_1793104252187053200/episode_000';window_states=np.load(proof/'integration_states.npz')['states'];assert np.array_equal(states,window_states)
    mirror=read(proof/'mirror_validation.json');assert mirror['samples']>0;assert all(x['qpos_max_difference']==0 and x.get('qvel_max_difference',0)==0 and x.get('ctrl_max_difference',0)==0 and x.get('platform_position_max_difference',0)==0 and x['owner_qpos_sha256']==x['display_qpos_sha256'] for x in mirror['rows'])
    frozen=read(ROOT/'results/phase1_5_frozen_history.json');changes=[p for p,m in frozen['files'].items() if not (ROOT/p).exists() or sha(ROOT/p)!=m['sha256']];assert not changes
    rows=[]
    for folder in sorted((ROOT/'results/runs').glob('phase1_5_session_*/episode_*')):
        if (folder/'summary.json').exists():
            summary=read(folder/'summary.json');entry=dict(folder=str(folder.relative_to(ROOT)),**{k:summary.get(k) for k in ['episode','controller','condition','physics_steps','simulation_time_s','wall_seconds','terminal_reason','failure','video_frames','mujoco_warnings']})
            if summary['physics_steps']:
                df=pd.read_csv(folder/'physics.csv.gz');assert len(df)==summary['physics_steps'];numeric=df.drop(columns=['failure']).select_dtypes(include='number');assert np.all(np.isfinite(numeric.to_numpy())),str(folder);entry.update(minimum_recorded_proxy_mm=float(df.proxy_clearance_m.min()*1000),max_position_error_mm=float(df.position_error_mm.max()),max_orientation_error_deg=float(df.orientation_error_deg.max()),max_force_N=float(df[[f'force_{i}' for i in range(6)]].abs().to_numpy().max()))
            rows.append(entry)
    pd.DataFrame(rows).to_csv(ROOT/'results/phase1_5_run_summary.csv',index=False)
    video_audit={}
    for name in ['stewart_phase1_5_teleop','stewart_phase1_5_dynamic_dashboard']:
        path=ROOT/'assets/videos'/f'{name}.mp4';audit=read(path.with_suffix('.frame_audit.json'));reader=imageio.get_reader(path);meta=reader.get_meta_data();count=0;distinct=0;previous=None
        for frame in reader:
            assert frame.shape==(1080,1920,3);sample=frame[::24,::24];distinct+=int(previous is not None and not np.array_equal(previous,sample));previous=sample.copy();count+=1
        reader.close();assert count==audit['frames'];assert meta['fps']==25;assert audit['no_future_curve_samples'];assert audit['all_state_hashes_match'];assert audit['qpos_range_max']>0;assert distinct>0
        video_audit[name]=dict(file=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,decoded_frames=count,resolution=list(meta['size']),fps=meta['fps'],duration_s=meta['duration'],all_frames_decoded=True,changed_decoded_frame_pairs=distinct,physical_qpos_range_max=audit['qpos_range_max'],no_future_curves=True,all_frame_state_hashes_match=True,source=audit['source'],source_kind=audit['source_kind'],source_physical_time_range_s=[audit['frame_mapping'][0]['time_s'],audit['frame_mapping'][-1]['time_s']],complete_frame_mapping=str(path.with_suffix('.frame_audit.json').relative_to(ROOT)))
    write_json(ROOT/'results/phase1_5_video_frame_audit.json',video_audit)
    # Dedicated high-resolution figures from complete real teleoperation logs.
    teleop=ROOT/'results/runs/phase1_5_session_1793093569084997300/episode_000';df=pd.read_csv(teleop/'physics.csv.gz').iloc[::10];t=df.time_s
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    colors=['#009add','#e99017','#36a94b','#e6547b','#946bdd','#998900']
    for j,ax in enumerate(axes.flat):
        ax.plot(t,df[f'q_ref_{j}']*1000,'--',color=colors[j],label='Reference');ax.plot(t,df[f'q_{j}']*1000,color=colors[j],label='Actual');ax.set_title(f'Leg {j+1} relative slide stroke');ax.set_ylabel('mm');ax.grid(alpha=.3);ax.legend()
    for ax in axes[-1]:ax.set_xlabel('Physical simulation time (s)')
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase1_5_six_slide_strokes.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for j,(ax,name) in enumerate(zip(axes,['Roll','Pitch','Yaw'])):
        ax.plot(t,df[f'rpy_ref_{j}'],'--',label='Reference');ax.plot(t,df[f'rpy_{j}'],label='Actual');ax.set_ylabel(name+' (deg)');ax.grid(alpha=.3);ax.legend()
    axes[-1].set_xlabel('Physical simulation time (s)');fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase1_5_platform_attitude.png',dpi=150);plt.close(fig)
    # Copy screenshots captured by this application's own Screenshot command.
    for folder,name in [('phase1_5_session_1793102606692461600/episode_001','phase1_5_control_panel.png'),('phase1_5_session_1793103154814442600/episode_000','phase1_5_failure_panel.png')]:
        files=sorted((ROOT/'results/runs'/folder).glob('dashboard_*.png'))
        if files:shutil.copy2(files[0],ROOT/'assets/plots'/name)
    sessions={}
    for n in ['1793102606692461600','1793103154814442600','1793103296667508700','1793103824929243500','1793104252187053200']:
        p=ROOT/'results/runs'/f'phase1_5_session_{n}';sessions[n]=read(p/'session.json');assert all(x.get('owner_exitcode')==0 and x.get('mirror_exitcode')==0 for x in sessions[n]['episodes'])
    fail=ROOT/'results/runs/phase1_5_session_1793103154814442600/episode_000';events=[json.loads(s) for s in (fail/'events.jsonl').read_text(encoding='utf-8').splitlines()];protection=next(e for e in events if e['type']=='protect_hold');failure_summary=read(fail/'summary.json')
    running=read(ROOT/'results/runs/phase1_5_session_1793103824929243500/episode_000/summary.json');assert running['terminal_reason']=='native_viewer_closed';assert running['failure'] is None
    junit=ET.parse(ROOT/'results/phase1_5_pytest.xml').getroot();suites=list(junit.iter('testsuite'));tests=sum(int(x.get('tests',0)) for x in suites);failures=sum(int(x.get('failures',0))+int(x.get('errors',0)) for x in suites)
    validation=dict(history=dict(files=frozen['count'],changed=changes),tests=dict(total=tests,failures=failures,report='results/phase1_5_pytest.xml'),recording_equivalence=backend['recording_equivalence'],window_equivalence=dict(reference=str(base.relative_to(ROOT)),window_run=str(proof.relative_to(ROOT)),physics_steps=len(states),state_width=states.shape[1],full_integration_states_bitwise_equal=True,max_difference=0.,same_terminal_reason=read(base/'summary.json')['terminal_reason']==read(proof/'summary.json')['terminal_reason'],mirror_samples=mirror['samples'],maximum_mirror_qpos_qvel_ctrl_platform_difference=0.),windows_acceptance=dict(keyboard_slider='Actual Windows keyboard W/A/I, slider inputs and physical logs retained',hold_home='Actual SPACE and R events in 1793102606692461600/episode_000',pause_close='Episode 001 C2 paused at 16.840s, panel Alt+F4, both processes exit 0',reset='C0->C2 uses new owner PID and episode',failure_close='Real asymmetric_gain protection, native window closed, both processes exit 0',running_close='Native standard Close button clicked during actual force execution and recording; both processes exit 0',native_key_focus='W in native Viewer produced no keyboard:W motion event in 1793103296667508700',replay='Actual Windows replay opened, Frame>/P/Play-Pause exercised; RGB source labelled ACTUAL EXECUTION CAPTURE',repeat_and_burst='Qt auto-repeat rejection tested; 100-target bounded pending-target stress test passed'),known_failed_attempts=['First owner smoke stopped at step 0 due to event argument name conflict; preserved and fixed','Initial freeze-manifest test used Windows GBK default; UTF-8 fixed, first test report preserved','First compositor used macroblock 16 and produced 1088p; attempt preserved, delivery re-encoded at exact 1080p','Alt+F4 attempt in running native Viewer did not establish closure; that run ended by wall budget. Subsequent direct system Close-button test passed'],protection=dict(first_violation_s=protection['time_s'],reason=protection['reason'],hold_end_s=failure_summary['simulation_time_s'],actual_hold_duration_s=failure_summary['simulation_time_s']-protection['time_s'],failure_preserved=True),sessions=sessions,video_audit='results/phase1_5_video_frame_audit.json',no_rl_training=True,limits=['Partial 86-pair proxy query, not complete robot collision qualification','Physics is slower than real-time with RGB recording; GUI may drop old display snapshots','Only Z/Roll/Pitch restricted workspace is open','Future RL telemetry schema exists but no RL integration/training is claimed'])
    write_json(ROOT/'results/phase1_5_interaction_validation.json',validation)
    report=f'''# Stewart Phase 1.5 验收报告

本阶段已完成受限交互、双窗口、真实执行记录、历史回放及同步视频。没有训练 PPO/BC/SAC，没有修改 Phase 0/Phase 1 控制器、模型或正式结果。工作目录：`__PROJECT_ROOT__`。

## 真实运行证据

- 自动测试：**{tests} 项，失败/错误 {failures} 项**；包含原 57 项回归，详见 `results/phase1_5_pytest.xml`。
- 冻结历史：{frozen['count']} 个旧文件逐一 SHA256 复核，变化 {len(changes)} 个。
- 录制开关：同一固定输入的 8000 步完整 integration state 逐位一致，最大差异 0。
- 最终双窗口：{len(states)} 步、每步 {states.shape[1]} 个 integration state 数值，与无窗口基线逐位一致；终止语义相同。
- 原生镜像：{mirror['samples']} 次样本，qpos/qvel/ctrl/平台位置最大差异 0；没有额外 mj_step。
- 失配诊断：首次保护 {protection['time_s']:.3f}s，原因 `{protection['reason']}`，保持至 {failure_summary['simulation_time_s']:.3f}s。不是静态安全停止，保持期间真实位姿继续变化，完整日志保留。
- 运动中关闭原生窗口：实际执行 {running['physics_steps']} 步 / {running['simulation_time_s']:.3f}s，保存 {running['video_frames']} 帧，归档原因 `{running['terminal_reason']}`；owner 与 mirror 退出码均 0。

本阶段交互/验收的全部 session 摘要见 `results/phase1_5_run_summary.csv`；固定预设及失配测试见 `phase1_5_backend_validation.json`。这些是界面验收，不替代或扩充旧 Phase 1 正式成功率。未筛掉失败尝试。

## 操作验收

Windows 实测了 W/A/I、Z 滑块、SPACE 真实状态保持、R 平滑 Home、P 暂停、Reset 新 episode 和 C2、原生窗口关闭、录制中关闭、暂停时关闭及失败后关闭。操作记录位于各 episode 的 events.jsonl；进程退出码位于 session.json。

原生窗口 W 不发送平台运动指令。自动重复键拒绝及 100 个连续目标的合并由新增自动测试覆盖；不把它们说成已经人工按住全部按键的穷尽测试。原生窗口 Q/Esc 另提供退出请求；本轮系统 Close 按钮已实测。所有测试只支持当前 Windows/驱动/模型条件，不保证任意远程桌面环境无故障。

历史回放已实际打开并操作逐帧/播放。现场 RGB 与仅状态重建明确区分，Replay 不额外执行物理步。日志缺少的旧内部求解器状态不被虚构。

## 视频与图表

| 文件 | 帧数 | 时长 | 分辨率 / 帧率 | 来源 |
|---|---:|---:|---|---|
| stewart_phase1_5_teleop.mp4 | {video_audit['stewart_phase1_5_teleop']['decoded_frames']} | {video_audit['stewart_phase1_5_teleop']['duration_s']}s | 1920×1080 / 25fps | Windows 键盘与滑块现场 RGB，截取物理时间9～34s |
| stewart_phase1_5_dynamic_dashboard.mp4 | {video_audit['stewart_phase1_5_dynamic_dashboard']['decoded_frames']} | {video_audit['stewart_phase1_5_dynamic_dashboard']['duration_s']}s | 1920×1080 / 25fps | C0 真实 Z 预设执行 |

两个视频所有帧均实际解码通过，画面有变化，真实 qpos 有位移；状态索引与哈希一致，曲线没有未来数据。逐帧证据见 `results/phase1_5_video_frame_audit.json` 及视频同名 `.frame_audit.json`。视频中的小运动未被人为放大。

截图：`assets/plots/phase1_5_native_window.png`、`phase1_5_control_panel.png`、`phase1_5_failure_panel.png`。曲线：`phase1_5_six_slide_strokes.png`、`phase1_5_platform_attitude.png`。截图只含项目窗口；其他应用的遮挡画面未作为交付截图。

## 最终十项回答

1. 可以用键盘控制上平台受限 Z/Roll/Pitch，须聚焦控制面板。
2. 可以用三个滑块设定目标；组合目标须通过受限椭球及完整 IK 预检。
3. 物理 owner 确实通过六执行器力输入和 mj_step 运动。原生 Viewer 是该真实状态的独立同步镜像，现场录像来自 live owner。
4. 六移动副相对行程可实时观察，支持单缸、参考/实际、误差、5/10/20秒。
5. Roll/Pitch/Yaw 均有参考和实际曲线，控制误差继续使用 SO(3)。
6. 可以按支链查看五个被动关节。
7. 两个最终视频同步展示机器人、真实曲线和状态。
8. 曲线时间来自 MuJoCo 物理时间，所有数据来自实际运行。
9. 当前固定输入的录制/窗口开关对照逐位一致；墙钟耗时会改变。该结论不推广到不同人工输入时刻。
10. 后续 Gymnasium/PPO 可以复用遥测和记录结构，但尚未接入 RL，当前奖励/策略字段为 N/A。

## 限制与未开展事项

保持 PARTIAL PROXY QUERY：86配对不等于完整碰撞安全。未开放 X/Y/Yaw，未作整机接触认证、连续碰撞证明或实机测试。GUI/视频是低速教学工具，非硬实时控制器。没有开展新算法、RL 训练或扰动自由遥操作。额外控制器对比视频为可选项，本轮未制作。

保留的失败尝试及修复经过见 `phase1_5_interaction_validation.json`。其中早期 Alt+F4 不能证明运动中原生关闭；最终用原生系统关闭按钮完成了独立验收，不能把前一次墙钟预算结束混记为关闭成功。

## 运行命令

```powershell
cd __PROJECT_ROOT__
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts\\teleop_stewart.py --record
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts\\view_stewart_dashboard.py --preset z --record --max-sim-seconds 6 --close-on-complete
& D:\\miniconda\\envs\\stewart_rl\\python.exe scripts\\replay_stewart.py results\\runs\\phase1_5_session_1793093569084997300\\episode_000
& D:\\miniconda\\envs\\stewart_rl\\python.exe -m pytest tests -p no:cacheprovider
```

这些入口及参数已实际运行；安装重建见文档08。操作说明见 `08_Phase1_5交互操作说明.md`，数据所有权与视频来源见 `09_Phase1_5实时曲线与视频设计.md`。

**完成 Phase 1.5 后暂停，等待审核，不自动开展 RL。**
'''
    (ROOT/'docs/10_Phase1_5验收报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(tests=tests,failures=failures,history_changes=changes,window_states_equal=True,videos={k:{x:v[x] for x in ['decoded_frames','resolution','duration_s']} for k,v in video_audit.items()}),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
