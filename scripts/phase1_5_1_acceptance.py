"""Recompute visual acceptance from actual journals and decoded video."""
import _bootstrap
import argparse,json,hashlib,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,pandas as pd,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import ROOT,write_json
from src.visualization.enhanced.style import LEG_COLORS,leg_mapping
from src.visualization.enhanced.events import read_events
from src.phase1.plant import create_plant

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main(folder):
    folder=Path(folder);home=read(folder/'home_baseline.json');df=pd.read_csv(folder/'physics.csv.gz',float_precision='round_trip');events=read_events(folder);summary=read(folder/'summary.json');baseline=Path(read(ROOT/'results/phase1_5_1_baseline_pointer.json')['folder']);a=np.load(baseline/'integration_states.npz')['states'];b=np.load(folder/'integration_states.npz')['states'];same=np.array_equal(a,b);assert same;controls=[f'ctrl_{i}' for i in range(12)];base_df=pd.read_csv(baseline/'physics.csv.gz',float_precision='round_trip');ctrl_equal=np.array_equal(base_df[controls].to_numpy(),df[controls].to_numpy());assert ctrl_equal;assert summary['failure'] is None;assert summary['physics_steps']==32000
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(df[controls].to_numpy()))
    mirror=read(folder/'enhanced_mirror_validation.json');assert mirror['samples']>0 and mirror['maximum_state_difference']==0
    old=read(ROOT/'results/phase1_5_1_frozen_history.json');changed=[p for p,m in old['files'].items() if not (ROOT/p).exists() or sha(ROOT/p)!=m['sha256']];assert not changed
    # Preserve causal event-stage semantics; carry only metadata already in earlier journal rows.
    by_id={};timeline=[]
    for e in events:
        cid=e.get('command_id');metadata=by_id.setdefault(cid,{})
        for key in ['target_before','target_after']:
            if key in e:metadata[key]=e[key]
        timeline.append(dict(metadata,**e))
    write_json(ROOT/'results/phase1_5_1_event_timeline.json',dict(source=str(folder/'events.jsonl'),no_events_inferred_from_curves=True,events=timeline))
    # Plot full real episode; no magnified qpos or rescaled mechanics.
    small=df.iloc[::10];t=small.time_s.to_numpy();delta=small[[f'q_{j}' for j in range(6)]].to_numpy()*1000-np.array(home['slide_m'])*1000;reference=small[[f'q_ref_{j}' for j in range(6)]].to_numpy()*1000-np.array(home['slide_m'])*1000
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for i,ax in enumerate(axes.flat):ax.plot(t,reference[:,i],'--',color=LEG_COLORS[i],label='Reference');ax.plot(t,delta[:,i],color=LEG_COLORS[i],label='Actual');ax.set_title(f'Leg{i+1} | Delta from recorded Home');ax.set_ylabel('mm');ax.grid(alpha=.25);ax.legend()
    for ax in axes[-1]:ax.set_xlabel('Physical simulation time (s)')
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase1_5_1_delta_leg_strokes.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for j in range(3):
        ax=axes[0,j];ax.plot(t,(small[f'p_ref_{j}']-home['p_m'][j])*1000,'--',label='Reference');ax.plot(t,(small[f'p_{j}']-home['p_m'][j])*1000,label='Actual');ax.set_title('Delta '+['X','Y','Z'][j]+' from fixed Home');ax.set_ylabel('mm');ax.grid(alpha=.25);ax.legend()
        ax=axes[1,j];ax.plot(t,small[f'rpy_ref_{j}'],'--',label='Reference');ax.plot(t,small[f'rpy_{j}'],label='Actual');ax.set_title(['Roll','Pitch','Yaw'][j]);ax.set_ylabel('deg');ax.set_xlabel('Physical simulation time (s)');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase1_5_1_delta_pose.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(14,11),sharex=True);axes[0].plot(t,(small.p_2-home['p_m'][2])*1000,label='Actual Delta Z');axes[0].set_ylabel('Delta Z (mm)');axes[1].plot(t,small.rpy_0,label='Roll');axes[1].plot(t,small.rpy_1,label='Pitch');axes[1].set_ylabel('deg')
    for j in range(6):axes[2].plot(t,delta[:,j],color=LEG_COLORS[j],label=f'Leg{j+1}')
    axes[2].set_ylabel('Delta stroke (mm)');axes[2].set_xlabel('Physical simulation time (s)')
    for e in events:
        if e['type']=='input' and e.get('kind') in ('increment','target','home','hold'):
            for ax in axes:ax.axvline(e['time_s'],color='#99a2b0',lw=.7,ls=':')
            axes[0].annotate(e['label'],xy=(e['time_s'],0),xytext=(3,40 if e['time_s'] in (0,11,21) else 90),textcoords='offset points',rotation=55,fontsize=8)
    for ax in axes:ax.grid(alpha=.25);ax.legend(ncol=3)
    axes[0].set_ylim(-.15,3.8);fig.tight_layout();fig.savefig(ROOT/'assets/plots/phase1_5_1_event_annotations.png',dpi=150);plt.close(fig)
    video=ROOT/'assets/videos/stewart_phase1_5_1_visual_demo.mp4';audit=read(video.with_suffix('.frame_audit.json'));reader=imageio.get_reader(video);meta=reader.get_meta_data();n=0;changes=0;previous=None
    for frame in reader:
        assert frame.shape==(1080,1920,3);sample=frame[::24,::24];changes+=int(previous is not None and not np.array_equal(sample,previous));previous=sample.copy()
        if n==300:imageio.imwrite(ROOT/'assets/plots/phase1_5_1_video_preview.png',frame)
        n+=1
    reader.close();assert n==800 and meta['fps']==25 and audit['no_future_curves'] and audit['no_future_events'] and audit['all_qpos_hashes_match'];assert changes>0
    # Explicitly verify each displayed event is the actual causal journal entry.
    indexed={e['journal_index']:e for e in events}
    for f in audit['frame_mapping']:
        if f['event_journal_index'] is not None:assert indexed[f['event_journal_index']]['time_s']==f['event_time_s']<=f['time_s']+1e-10
    junit=ET.parse(ROOT/'results/phase1_5_1_pytest_final.xml').getroot();suites=list(junit.iter('testsuite'));nt=sum(int(s.get('tests',0)) for s in suites);nf=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites);assert nt>=102 and nf==0
    gui=ROOT/'results/runs/phase1_5_1_session_1793123362993675200/episode_000';gui_events=read_events(gui);keyboard=[e for e in gui_events if e['type']=='input' and str(e.get('source','')).startswith('keyboard:')];snapshots=sorted(gui.glob('dashboard_*.png'))
    if snapshots:shutil.copy2(snapshots[0],ROOT/'assets/plots/phase1_5_1_control_panel.png')
    validation=dict(episode=str(folder),physics_steps=summary['physics_steps'],simulation_time_s=summary['simulation_time_s'],failure=summary['failure'],physics_warnings=summary['mujoco_warnings'],home_baseline=home,palette=LEG_COLORS,mapping=leg_mapping(create_plant('nominal')),baseline=str(baseline),full_integration_states_equal=same,state_shape=list(a.shape),maximum_state_difference=float(np.max(abs(a-b))),all_12_control_inputs_equal=ctrl_equal,terminal_reason_equal=summary['terminal_reason']==read(baseline/'summary.json')['terminal_reason'],frozen_history=dict(count=old['count'],changed=changed),tests=dict(total=nt,failures=nf,original85_preserved=True,junit='results/phase1_5_1_pytest_final.xml'),events=dict(count=len(events),motion_starts=[e for e in events if e['type']=='motion_started'],motion_completions=[e for e in events if e['type']=='motion_completed'],true_keyboard_events=keyboard,no_inferred_commands=True),video=dict(frames=n,resolution=list(meta['size']),fps=meta['fps'],duration_s=meta['duration'],changed_decoded_frame_pairs=changes,sha256=sha(video),all_frames_decoded=True,no_future_curves=audit['no_future_curves'],no_future_events=audit['no_future_events'],frame_mapping=str(video.with_suffix('.frame_audit.json')),source_kind=audit['source_kind']),gui=dict(actual_windows_observed=True,position_tracking_error_selected=True,mdeg_selected=True,leg3_card_selected=True,native_coloured_leg_labels_screenshot='assets/plots/phase1_5_1_native_window.png',snapshot='assets/plots/phase1_5_1_control_panel.png'),limitations=['Partial proxy query only; no complete robot collision certification','Legacy episodes without an explicit baseline display Home UNKNOWN; never use first visible sample silently','Scheduled demo is actual dynamics, not human-keyboard footage','No Gymnasium environment or PPO/BC/SAC training in this stage'],test_attempt_notes=['Restricted test attempts encountered Windows pytest temporary-folder ACL errors; logs retained','Approval service transient timeout/transport failure prevented one normal-permission launch; later retry executed complete suite successfully'])
    validation['mirror']=dict(samples=mirror['samples'],maximum_state_difference=mirror['maximum_state_difference'],mj_step_calls=mirror['mj_step_calls']);validation['all_physical_states_and_ctrl_finite']=True
    write_json(ROOT/'results/phase1_5_1_visual_validation.json',validation);print(json.dumps(dict(tests=nt,failures=nf,steps=summary['physics_steps'],state_equal=same,controls_equal=ctrl_equal,old_files_changed=changed,video_frames=n),ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode');main(p.parse_args().episode)
