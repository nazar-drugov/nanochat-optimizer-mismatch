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
