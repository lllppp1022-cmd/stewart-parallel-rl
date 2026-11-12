"""Generate report only from completed runs and independently recomputed metrics."""
import _bootstrap,json,platform
from pathlib import Path
import numpy as np,pandas as pd
from research_phase2c2 import RUN,PROTOCOL
from src.model import ROOT,write_json,sha

def main():
    d=pd.read_csv(ROOT/'results/phase2c2_evaluation_comparison.csv');t=pd.read_csv(ROOT/'results/phase2c2_training_comparison.csv');gate=json.loads((RUN/'expansion_gate.json').read_text(encoding="utf-8"));video=json.loads((ROOT/'results/phase2c2_recording_validation.json').read_text(encoding="utf-8"));hist=json.loads((RUN/'history_preservation.json').read_text(encoding="utf-8"));audit=json.loads((RUN/'training_journal_audit.json').read_text(encoding="utf-8"));obs=json.loads((RUN/'observer_equivalence_v2.json').read_text(encoding="utf-8"));reg=json.loads((RUN/'regression_results.json').read_text(encoding="utf-8"));ladder=json.loads((RUN/'amplitude_ladder.json').read_text(encoding="utf-8"));diags=json.loads((RUN/'action_diagnostics.json').read_text(encoding="utf-8"));testlog=(ROOT/'results/phase2c2_final_tests.log').read_text(encoding='utf-8')
    assert audit['passed'] and hist['passed'] and video['passed'] and obs['passed'];sums=[]
    for f in RUN.rglob('summary.json'):
        x=json.loads(f.read_text(encoding="utf-8"));sums.append(dict(folder=str(f.parent),steps=x.get('physics_steps',0),decisions=x.get('rl_decisions',0),reason=x.get('terminal_reason'),failure=x.get('failure'),complete=x.get('complete')))
    totalsteps=sum(x['steps'] for x in sums);write_json(RUN/'interaction_inventory.json',dict(total_physical_steps_in_run=totalsteps,episodes=sums,note='Includes ladder, action probes, baselines, observer PPO diagnostics, formal PPO, stage validation, final and Home regression. Pytest temp trials separately excluded.'))
    base=d[d.label=='four_d_lead'].z_rmse_mm.mean();table='| 方法 | 到达 | Z RMSE均值(mm) | 相对强基线 | XY峰值均值(mm) | 最小代理间隙(mm) |\n|---|---:|---:|---:|---:|---:|\n'
    for method in ['c4','fz_lead','four_d_lead','PPO42','PPO123']:
        g=d[d.label==method];z=g.z_rmse_mm.mean();table+=f'| {method} | {int(g.success.sum())}/{len(g)} | {z:.9f} | {(z/base-1)*100:+.3f}% | {g.xy_peak_mm.mean():.6f} | {g.minimum_partial_proxy_m.min()*1000:.5f} |\n'
    trajectorytable='| 参考幅值 | 工况 | 实际Z范围(mm) | Z RMSE(mm) | 完成 |\n|---|---|---:|---:|---|\n'
    for r in ladder:trajectorytable+=f'| {r["task"]["amplitude_m"]*1000:.0f}mm | tau={r["task"]["tau_s"]}s, Fz={r["task"]["Fz_N"]}N | {r["z_ptp_mm"]:.6f} | {r["z_rmse_mm"]:.6f} | {r["success"]} |\n'
    trainlines='\n'.join(f'- seed{int(r.seed)}：{int(r.decisions)} PPO决策，{int(r.physical_steps)}物理积分步，{int(r.updates)}次rollout更新，{int(r.optimizer_steps)}个Adam步，{int(r.complete_episodes)}个完整回合。' for r in t.itertuples())
    warnings=int(d.warning_count.sum());contacts=int(d.native_contact_observations.sum());jgate=json.loads((RUN/'joint_2048_gate.json').read_text(encoding="utf-8"));stop='达到4096共同预算' if t.decisions.min()==4096 else '2048共同门禁未通过，两个种子均停止';obs_budget=384;decision=int(t.decisions.sum())
    test_summary=next(line for line in reversed(testlog.splitlines()) if 'passed' in line and 'xfailed' in line)
    conclusion='满足进一步制定扩大研究协议的条件；本报告没有把未运行8192/16384写成完成。' if gate['passed'] else '未通过扩大预算门禁。完成本轮有限PPO研究后停止，不开展8192/16384或100k训练。'
    regtable='| 旧Home回归方法 | Z RMSE(mm) | 相对强基线 | 到达 |\n|---|---:|---:|---|\n'
    for rr in reg:
        name='four_d_lead' if 'training_seed' not in rr else 'PPO'+str(rr['training_seed']);regtable+=f'| {name} | {rr["z_rmse_mm"]:.9f} | {(rr["z_rmse_mm"]/reg[0]["z_rmse_mm"]-1)*100:+.3f}% | {rr["success"]} |\n'
    pooled_z=d[d.label.str.startswith('PPO')].z_rmse_mm.mean();benefit=(1-pooled_z/base)*100
    report=f'''# Phase2C-2动态任务与残差PPO验收报告

日期：2026-10-30。实验目录：`{RUN}`。本报告指标由真实CSV重新计算，不引用旧实验数字冒充新结果。

## 结论

已将固定Home任务扩展为先2毫米、再3毫米的真实往返升降；没有修改Home±5毫米保护边界。已完成动作诊断、强基线对照、真实PPO两种子有限训练、冻结后的新任务评价、旧Home回归和现场录像。

{conclusion}

两条新最终任务上的结果如下。两个测试任务和两个训练种子均不足以宣称统计显著或广泛泛化。完整每任务结果见`results/phase2c2_evaluation_comparison.csv`。

{table}

Z主指标是整个六秒轨迹误差RMSE，不是仅选择加载窗口或成功片段。主基线为原C4+fourDlead，所有方法使用相同路径、物理滞后、负载、限制与结束标准。

合并两个训练种子，平均改善{benefit:.3f}%，低于预冻结10%门槛；绝对平均改善{(base-pooled_z)*1000:.3f}微米。存在方向一致的小幅学习收益，但不足以支持直接增加8192/16384预算或宣称已经充分解决未知动力学问题。旧Home精度代价也必须保留。

## 真实位移与动作诊断

{trajectorytable}

24个动作/频率诊断全部保存：三档滞后、20/50Hz决策、零/正/负/正弦动作。完成{sum(r['success'] for r in diags)}/24，未通过则不会进入训练。3毫米参考下实际范围{min(r['z_ptp_mm'] for r in diags):.6f}–{max(r['z_ptp_mm'] for r in diags):.6f}毫米。

按上升和回落分别计算，正力补偿可以改善上升、负力补偿可以改善回落，反向会恶化；常量动作不适合整个往返轨迹。分段证据见`results/phase2c2_action_response.csv`。50Hz没有显示明确的局部测试优势，因此仍用20Hz完成这轮学习；这不是20/50Hz PPO收敛能力的正式比较。

无负载/名义模型在运动时也有跟踪误差。该部分不能归因于未知滞后；现有基线可能仍可通过传统参考前馈进一步增强。本轮不通过削弱它制造RL收益。Fz-only lead与fourDlead在此单Z任务很接近，这与其他方向激励很小相符。

## 真实训练成本与共同门禁

{trainlines}

正式合计{decision}决策、{int(t.physical_steps.sum())}物理步、{int(t.updates.sum())}次PPO更新。{stop}。256烟雾是累计预算前缀，没有额外给某个种子训练。2048门禁共同结果：{jgate['passed']}。具体预冻结阈值与每个阶段结果保留在`joint_*_gate.json`和各种子`stage_*.json`。

另有观察器诊断{obs_budget}决策/19200物理步（关闭128、开启128、调整相机后开启128），不混入正式样本效率。新run目录所有实际执行合计{totalsteps}物理步，包括诊断、评价、保护保持及中途预算截断，详细清单见`interaction_inventory.json`。pytest临时目录的验证步数不计为研究训练。

主训练所有Actor/Critic更新范数非零，模型逐阶段保存并重载参数完全相等；每256决策保存真实rollout、GAE、价值目标与checkpoint SHA。原始动作和采样日志逐一核对，最大重算奖励误差{t.max_reward_recompute_error.max():.3g}。奖励被上限截断的物理步占比最高{t.reward_cap_fraction.max():.3g}，没有用奖励饱和掩盖改善。网络结构和奖励尺度见文档34及各种子配置。

## 安全、模型隔离与公平性

最终评价10个回合完成{int(d.success.sum())}/10，MuJoCo warning记录{warnings}，原生接触累计观察{contacts}。正式训练保护回合数{int(t.protected_episodes.sum())}。全部关节状态由原真实积分推进，未在训练中写live qpos。

最小间隙属于现有**部分代理几何**距离，不能等同完整碰撞认证；20ms代理查询也不等同连续安全保证。执行器滞后是合成一阶动态，不是已辨识液压缸模型。控制器/Actor不读取真实tau或负载时间，理想关节反馈限制保留。

零RL在运动与加载的2200步测试中，与原fourDlead的qpos/qvel/ctrl逐位相同。合成残差仍受原共享±5N/20N/s及原生±400N限制。原78/85维Panda策略与本项目无关，本阶段没有BC。

## 旧任务回归与视频

旧Home回归保留在`regression_results.json`，共{len(reg)}回合，完成{sum(r['success'] for r in reg)}/{len(reg)}；它是已经见过的历史条件，不是新的最终测试。

{regtable}

回归任务到达不等于精度未退化。若表中PPO的Home误差增加，必须与动态任务改善一起报告，不能只报告成功率。

视频：`assets/videos/phase2c2_actual_training_visible_v2.mp4`，{video['decoded_frames']}帧、{video['playback_seconds']:.0f}秒、25fps、1920×1088。来自三条各6秒的真实训练现场RGB，4倍慢放和4倍像素裁剪明确标注；{sum(x['frames'] for x in video['sources'])}个原始RGB帧的qpos哈希与全状态日志一致。不存在重新确定性执行冒充训练。

录像开/关对照的样本、观测、奖励、GAE、结束类型和最终网络参数完全一致；新相机重跑也通过。Windows回放窗口真实运行并自动播放/关闭，证据在`results/phase2c2_replay_*playing.png`。实时GUI附着到训练进程没有在本轮实现或验证；采用现场录像加独立回放。

原物理Owner局部episode字段为0，这一历史接口限制在原RGB/物理CSV仍可见；全局回合应使用`training_samples.csv`与目录编号。新回放顶部和综合视频明确使用目录全局编号，原数据未改写。

记录完成后已同步新环境Owner的全局回合编号，并对后续重置进行测试；这只修正元数据，不改变控制、奖励或观测。当前代码的这项显示修复晚于本轮实际训练，历史原数据仍使用旧局部编号。分析脚本曾将`episode_summary.json`误识别为回合目录，修复为只扫描目录后重新核验；首次报错日志也保留，未丢失训练结果。

## 文件冻结、测试与剩余事项

旧文件检查{hist['files_checked']}个，哈希变化{len(hist['changed'])}个。保留Phase2C-1原模型、结果、配置、视频和原始失败/负结果。新模型在`models/phase2c2/phase2c2_20261030_v1`。

最终pytest真实输出：`{test_summary}`，保留在`results/phase2c2_final_tests.log`。xfail是Phase0历史“不安装学习库”的规则，后续阶段已明确授权，不能计为普通通过。

未执行：8192/16384/100k扩展训练、三种子泛化矩阵、50/100Hz PPO等物理曝光对照、四/六维Actor、实时PPO GUI线程集成、真实硬件部署、完整碰撞认证。没有把这些工作宣称完成。

当前最合理下一步取决于预冻结门禁，而不是看到平台运动就继续加预算。本轮学习收益、名义精度、安全性及成本必须一起判断。扩大几何运动确实改善了可见性和任务对齐，但不自动意味着RL能够超过传统控制。

观看和PowerShell命令详见`docs/36_Phase2C2训练与运动观看教程.md`。本轮结束后暂停。
'''
    (ROOT/'docs/35_Phase2C2有限强化学习验收报告.md').write_text(report,encoding='utf-8')
    files={}
    paths=list(RUN.rglob('*'))+list((ROOT/'models/phase2c2'/RUN.name).rglob('*'))+list((ROOT/'docs').glob('3[456]_Phase2C2*'))+list((ROOT/'src/rl').glob('*phase2c2*'))+[ROOT/'src/rl/moving_reference.py']+list((ROOT/'scripts').glob('*phase2c2*.py'))+list((ROOT/'tests').glob('*phase2c2*.py'))+list((ROOT/'assets/plots').glob('phase2c2*'))+list((ROOT/'assets/videos').glob('phase2c2*'))
    for f in paths:
        if f.is_file() and '__pycache__' not in f.parts:files[str(f.relative_to(ROOT))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    write_json(ROOT/'results/phase2c2_artifact_manifest.json',dict(count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),files=files))
    print(table);print('EXPANSION',gate);print('TOTAL PHYSICAL',totalsteps)
if __name__=='__main__':main()
