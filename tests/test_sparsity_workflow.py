import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_sparsity_workflow as w


def test_ft_edges_extend_each_pair_and_dataset_independently():
    def analysis(pre, ft, lrs, best):
        rows = [dict(base_run=f'e4-main-{pre}-s42', ft_optimizer=ft, lr=lr) for lr in lrs]
        return {'lr_selection': rows, 'edge_winners': [r for r in rows if r['lr'] == best]}
    a = analysis('adamw','muon',[.000125,.00025,.0005],.0005)
    b = analysis('adamw','muon',[.000125,.00025,.0005],.000125)
    c = analysis('muon','adamw',[.001,.002,.004],.004)
    assert w.ft_extensions([('starcoderdata',a),('starcoderdata',b),('starcoderdata',a),('wikitext103',c)],
                           {'adamw':{'full_lr':.004},'muon':{'full_lr':.0005}})=={
        ('starcoderdata','adamw','muon'):[.125,2.],('wikitext103','muon','adamw'):[2.]}


@pytest.mark.parametrize('experts',[1,4,16])
@pytest.mark.parametrize('opt',['adamw','muon'])
def test_smoke_job_verification_only(experts,opt,tmp_path):
    script = w.smoke_job(experts,opt,tmp_path,tmp_path/'scratch')
    assert f'optimizer-mismatch-e{experts}/pretrain/verification' in script
    assert f'e{experts}-smoke-{opt}-s42' in script
    assert '--train-tokens=262144' in script and '--eval-every=1' in script
    assert ('--disable-routing-loss' in script)==(experts==1)
    p=tmp_path/'job.sbatch';p.write_text(script)
    subprocess.run(['bash','-n',str(p)],check=True)


def test_wave_durable_ledger_and_recovered_submission(monkeypatch,tmp_path):
    cfg=SimpleNamespace(experts=1,work_dir=tmp_path/'work',scratch_base=tmp_path/'scratch',poll_seconds=10,nice=0)
    c=w.Coordinator(cfg)
    submissions=[]
    def fake(args,**kw):
        if args[0]=='sbatch':
            submissions.append(args);return '12345;cluster\n'
        if args[0]=='squeue':return ''
        if args[0]=='sacct' and '--name' in args:return ''
        if args[0]=='sacct':return '12345|COMPLETED|0:0|\n'
        raise AssertionError(args)
    monkeypatch.setattr(w,'execute',fake)
    def generate(d): (d/'trial.sbatch').write_text('#!/bin/bash\n')
    c.wave('initial',generate,wait=False)
    assert len(submissions)==1
    ledger=json.loads((cfg.work_dir/'initial/jobs.json').read_text())
    assert [v['job_id'] for v in ledger.values()]==['12345']
    assert '--no-requeue' in submissions[0] and 'mit_normal_gpu' in submissions[0]
    c.wave('initial',lambda d: pytest.fail('must not regenerate a recorded wave'))
    assert len(submissions)==1
    monkeypatch.setattr(c,'recover_job',lambda name:'12345')
    c.wave('recovered', generate, filter_scripts=lambda p:True)
    assert len(submissions)==1


def test_hold_ft_pins_bases_without_generating_finetune_jobs(monkeypatch,tmp_path):
    cfg=SimpleNamespace(experts=1,work_dir=tmp_path/'work',scratch_base=tmp_path/'scratch',poll_seconds=10,nice=0,hold_ft=True)
    c=w.Coordinator(cfg)
    for opt in w.OPTS:
        name=f'e1-smoke-{opt}-s42'; p=c.checkpoints/name/'result.json'; p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(status='completed',steps_trained=1,run=name,tokens_trained=262144,
                                     final_val_bpb=3.,final_heldout_bpb=3.,final_train_loss=10.,
                                     settings=dict(optimizer_name=opt,seed=42,n_experts=1,train_tokens=262144,
                                                   total_batch_size=262144,disable_routing_loss=True,balance_coef=0.,z_coef=0.))))
    calls=[]
    def module(name,args,allowed=(0,)):
        calls.append((name,args)); return 0
    monkeypatch.setattr(c,'module',module)
    monkeypatch.setattr(c,'wave',lambda *a,**kw:None)
    c.run()
    assert c.state['phase']=='ft-hold'
    assert any('base-manifest' in args for _,args in calls)
    assert not any(name=='scripts.finetune_jobs' for name,_ in calls)
    c.hold_ft=False
    with pytest.raises(RuntimeError,match='retention policy'):
        c.run()
    assert not any(name=='scripts.finetune_jobs' for name,_ in calls)
    c.checkpoint_retention='full'
    c.state['checkpoint_retention']='weights'
    with pytest.raises(RuntimeError,match='conflicts with existing'):
        c.run()


def test_soft_router_storage_estimate_and_actual_quota_guard(monkeypatch,tmp_path):
    text='SCRATCH | 414.1 | 1024.0 | 40.44\nPOOL | 48.0 | 1024.0 | 4.69\n'
    assert w.quota_free_gb(text)==pytest.approx({'SCRATCH':609.9,'POOL':976.})
    with pytest.raises(ValueError,match='both'):
        w.quota_free_gb('SCRATCH | 400 | 1024')
    req=w.storage_requirements_gb(16,120,True,8192,'logits','none')
    assert req['SCRATCH']==req['POOL'] and 81<req['SCRATCH']<100
    full=w.storage_requirements_gb(16,120,True,8192,'logits','full')
    assert full['SCRATCH']>14000 and full['POOL']==req['POOL']
    cfg=SimpleNamespace(experts=16,work_dir=tmp_path/'work',scratch_base=tmp_path/'scratch',poll_seconds=10,nice=0,
                        checkpoint_retention='none',soft_router_capture=True,soft_router_sample_size=8192,soft_router_sample_format='logits')
    c=w.Coordinator(cfg)
    quota=tmp_path/'orcd'/'.quota'; quota.parent.mkdir(); quota.write_text(text)
    monkeypatch.setattr(w.Path,'home',lambda:tmp_path)
    c.check_ft_storage(120)
    quota.write_text('SCRATCH | 900 | 1024\nPOOL | 48 | 1024\n')
    c.check_ft_storage(120,'ft-initial')
    other=w.Coordinator(SimpleNamespace(experts=4,work_dir=tmp_path/'other',scratch_base=tmp_path/'scratch',
                                         poll_seconds=10,nice=0,checkpoint_retention='none',soft_router_capture=True,
                                         soft_router_sample_size=8192,soft_router_sample_format='logits'))
    with pytest.raises(RuntimeError,match='reserved'):
        other.check_ft_storage(120,'ft-initial')
    c.release_ft_storage('ft-initial')
    # Completed bytes may still be absent from the old report, so retain the
    # reservation until that report has actually been refreshed.
    with pytest.raises(RuntimeError,match='reserved'):
        other.check_ft_storage(120,'ft-initial')
    import os,time
    os.utime(quota,(time.time()+1,time.time()+1))
    other.check_ft_storage(120,'ft-initial')
    other.release_ft_storage('ft-initial')
    quota.write_text('SCRATCH | 1000 | 1024\nPOOL | 48 | 1024\n')
    with pytest.raises(RuntimeError,match='SCRATCH quota'):
        c.check_ft_storage(120)


def test_gpu_canary_evidence_gate_accepts_valid_hardlinked_arrays_rejects_corruption(tmp_path):
    import hashlib, os
    import numpy as np
    from nanochat.soft_routing import schema, METRICS, SOFT_ROUTER_SCHEMA_VERSION
    from scripts.finetune import checkpoint_steps
    root=tmp_path/'ft'; dataset=root/'e1-main-adamw-s42'/'starcoderdata'; base=dataset/'base';base.mkdir(parents=True)
    pos=np.arange(4,dtype=np.int64); digest=hashlib.sha256(pos.astype('<i8').tobytes()).hexdigest()
    counts=np.array([4,4]); L=2; M=len(METRICS); valid=np.ones(M,dtype=bool);valid[1]=False
    base_arrays={}; arrays={'soft_router_schema_version':np.array(SOFT_ROUTER_SCHEMA_VERSION),'layers':np.arange(L)}
    for prefix in ('d0','d1'):
        edges=np.tile(np.linspace(0,1,33),(M,1));routes=np.zeros((L,2,4),dtype=np.uint8)
        base_arrays.update({f'{prefix}_soft_sample_positions':pos,f'{prefix}_soft_sample_doc_ids':np.zeros(4,dtype=np.int64),
                            f'{prefix}_soft_doc_counts':counts,f'{prefix}_soft_sample_positions_sha256':np.array(digest),
                            f'{prefix}_routes':routes,f'{prefix}_soft_hist_edges':edges})
        hist=np.zeros((L,M,32),dtype=np.int64);hist[:,:,0]=np.where(valid,8,0)
        doc_sum=np.zeros((L,2,M));doc_sum[:,:,[2,3]]=4.
        vals={'soft_sample_positions_sha256':np.array(digest),'soft_doc_counts':counts,'soft_metric_valid':valid,
              'soft_layer_count':np.broadcast_to(np.where(valid,8,0),(L,M)), 'soft_hist_count':hist,'soft_hist_edges':edges,
              'soft_doc_sum':doc_sum, 'soft_doc_sumsq':doc_sum,
              'soft_layer_sum':doc_sum.sum(axis=1), 'soft_layer_sumsq':doc_sum.sum(axis=1),
              'soft_prob_sum':np.full((L,1),8.),'soft_doc_prob_sum':np.full((L,2,1),4.),
              'soft_sample_logits':np.zeros((L,4,1),dtype=np.float32),'soft_sample_probs':np.empty((L,0,1),dtype=np.float32),'routes':routes}
        arrays.update({f'{prefix}_{k}':v for k,v in vals.items()})
    np.savez_compressed(base/'base_arrays.npz',**base_arrays)
    shared={}
    for horizon in (60,240,960):
        run=dataset/f'adamw-lr0.001-s1-T{horizon}';run.mkdir()
        cell=dict(base_run='e1-main-adamw-s42',pretrain_optimizer='adamw',ft_dataset='starcoderdata',
                  ft_optimizer='adamw',lr=.001,seed=1,ft_steps=horizon,n_experts=1)
        decay=round(.2*horizon);steps=checkpoint_steps(horizon,decay)
        (run/'result.json').write_text(json.dumps(dict(status='completed',steps_completed=horizon,**cell,
             checkpoint_retention='none',attempt_id='same',curve=[{'step':step} for step in steps],soft_router=schema(4,'logits'))))
        (run/'config.json').write_text(json.dumps(dict(cell=cell,checkpoint_retention='none',attempt_id='same',
             warmup_steps=8,decay_steps=decay,soft_router=schema(4,'logits'))))
        for step in steps:
            path=run/f'ckpt_{step:06d}.npz'
            if step<=horizon-decay and step in shared: os.link(shared[step],path)
            else:
                np.savez_compressed(path,**arrays)
                if step<=horizon-decay: shared[step]=path
    result=w.verify_soft_canary(root,1,.001,4,'logits','none')
    assert result['unique_npz']==18 and result['npz_bytes']>0
    wrong={**arrays,'d0_soft_sample_logits':np.zeros((L,3,1),dtype=np.float32)}
    np.savez_compressed(dataset/'adamw-lr0.001-s1-T960'/'ckpt_000960.npz',**wrong)
    with pytest.raises(RuntimeError,match='logits sample'):
        w.verify_soft_canary(root,1,.001,4,'logits','none')
