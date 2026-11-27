"""Resume-safe scheduler coordinator for the approved E=1/4/16 seed-42 pilots.

Run from a pinned checkout on an allocated CPU node. The coordinator submits
only reviewed, generated training jobs, waits for successful Slurm accounting,
then advances the independent LR selections. Automatic extensions are exactly
factor-of-two bracket extensions. No matched-learning target is selected here.
One work directory per E/owner; do not launch multiple coordinators for it.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from nanochat.pretrain_sweep import render_job, sweep_runs


OPTS = ('adamw', 'muon')
HORIZONS = {'starcoderdata': (60, 240, 960), 'wikitext103': (60, 240, 720)}


def load(path):
    return json.loads(Path(path).read_text())


def store(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def execute(args, *, allowed=(0,), capture=False):
    print('+', ' '.join(map(str, args)), flush=True)
    r = subprocess.run(list(map(str, args)), text=True, capture_output=capture)
    if r.returncode not in allowed:
        raise RuntimeError(f'command failed ({r.returncode}): {args[0]}: {r.stderr or ""}')
    return r.stdout if capture else r.returncode


def ft_extensions(analyses, selection):
    """Aggregate edge winners across lengths, retaining the original 2x grid."""
    extensions = {}
    for dataset, analysis in analyses:
        for edge in analysis.get('edge_winners', []):
            pre = edge['base_run'].split('-')[2]
            ft = edge['ft_optimizer']
            if pre not in OPTS or ft not in OPTS:
                raise ValueError('unexpected optimizer in edge winner')
            grid = [r['lr'] for r in analysis['lr_selection'] if r['base_run'] == edge['base_run'] and r['ft_optimizer'] == ft]
            lr = edge['lr']
            if lr not in (min(grid), max(grid)) or len(grid) < 2:
                raise ValueError('edge winner does not match its grid')
            relative = lr / selection[ft]['full_lr']
            exponent = round(math.log2(relative))
            if not math.isclose(relative, 2. ** exponent, rel_tol=1e-12):
                raise ValueError('non-power-of-two fine-tuning grid; review rather than guess')
            factor = 2. ** (exponent + (-1 if lr == min(grid) else 1))
            if not math.isfinite(factor) or factor <= 0:
                raise ValueError('invalid extension factor')
            extensions.setdefault((dataset, pre, ft), set()).add(factor)
    return {k: sorted(fs) for k, fs in extensions.items()}


def edge_args(extras, flag):
    return [flag, *[f'{o}={x!r}' for o in OPTS for x in sorted(extras[o])]] if any(extras.values()) else []


class SmokeRun:
    """One real training step and two small evaluations, never an LR trial."""
    def __init__(self, experts, optimizer):
        grid = [r for r in sweep_runs(n_experts=experts) if r.optimizer == optimizer]
        self.spec = grid[len(grid)//2]
        self.name = f'e{experts}-smoke-{optimizer}-s42'

    def base_train_args(self):
        overrides = {'--train-tokens':262144, '--warmup-steps':0, '--warmdown-ratio':0,
                     '--eval-every':1, '--eval-tokens':131072, '--heldout-eval-tokens':131072, '--save-every':-1}
        args = [a for a in self.spec.base_train_args() if a.split('=')[0] not in overrides]
        return [*args, *[f'{k}={v}' for k,v in overrides.items()]]


def smoke_job(experts, optimizer, repo, scratch):
    run = SmokeRun(experts, optimizer)
    script = render_job(run.name, [run], '00:45:00', repo, scratch)
    env = (f'export WANDB_RUN_GROUP=optimizer-mismatch-e{experts}/pretrain/verification\n'
           f'export WANDB_JOB_TYPE=pretrain-verification\n'
           f'export WANDB_TAGS=optimizer-mismatch,E{experts},pretrain,verification,{optimizer},base-seed-42\n')
    marker = 'LAUNCHER="${NANOCHAT_LAUNCHER:-runs/pretrain_study.sh}"'
    if marker not in script: raise ValueError('generated launcher marker changed')
    return script.replace(marker, env + marker, 1)


class Coordinator:
    def __init__(self, args):
        self.e = args.experts
        self.work = args.work_dir.resolve()
        self.scratch = args.scratch_base.resolve()
        self.repo = Path(__file__).resolve().parents[1]
        self.checkpoints = self.scratch / 'base_checkpoints'
        self.ft_root = self.scratch / f'finetune_pilot_e{self.e}'
        self.poll = args.poll_seconds
        self.nice = args.nice
        self.work.mkdir(parents=True, exist_ok=True)
        self.selection = self.work / 'selection'; self.selection.mkdir(exist_ok=True)
        self.id = hashlib.sha256(str(self.work).encode()).hexdigest()[:10]
        self.state_path = self.work / 'state.json'
        if self.state_path.exists():
            self.state = load(self.state_path)
            if self.state['experts'] != self.e or self.state['scratch'] != str(self.scratch):
                raise ValueError('incompatible existing workflow state')
        else:
            self.state = {'experts': self.e, 'scratch': str(self.scratch), 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'phase': 'startup'}
            store(self.state_path, self.state)

    def phase(self, name, **details):
        if name != 'blocked': self.state.pop('error', None)
        self.state.update(phase=name, updated=time.strftime('%Y-%m-%dT%H:%M:%S'), **details)
        store(self.state_path, self.state)
        print(f'E={self.e}: {name}', flush=True)

    def module(self, mod, args, allowed=(0,)):
        return execute([sys.executable, '-m', mod, *args], allowed=allowed)

    def recover_job(self, name):
        # If the process died between sbatch and ledger write, discover the unique
        # deterministic job name in both queue and accounting before resubmitting.
        active = execute(['squeue', '-h', '-u', os.environ['USER'], '--name', name, '-o', '%i|%j'], capture=True)
        history = execute(['sacct', '-n', '-X', '-P', '-u', os.environ['USER'], '-S', self.state['created'],
                           '--name', name, '--format=JobIDRaw,JobName%100'], capture=True)
        ids = {line.split('|')[0] for line in (active+'\n'+history).splitlines()
               if len(line.split('|')) >= 2 and line.split('|')[1] == name and line.split('|')[0].isdigit()}
        if len(ids) > 1:
            raise RuntimeError(f'ambiguous prior submission for {name}: {sorted(ids)}')
        return next(iter(ids), None)

    def wave(self, label, generation, *, filter_scripts=None, wait=True):
        directory = self.work / label; directory.mkdir(parents=True, exist_ok=True)
        ledger_path = directory / 'jobs.json'
        if ledger_path.exists():
            jobs = load(ledger_path)
        else:
            generation(directory)
            scripts = sorted(directory.glob('*.sbatch'))
            if filter_scripts is not None:
                scripts = [p for p in scripts if filter_scripts(p)]
            jobs = {str(p): {'job_id': None, 'submit_intent': False} for p in scripts}
            store(ledger_path, jobs)
        for path, record in jobs.items():
            if record['job_id'] is not None:
                continue
            token = hashlib.sha256(f'{label}/{Path(path).name}'.encode()).hexdigest()[:8]
            name = f'e{self.e}-{label[:23]}-{self.id}-{token}'
            job = self.recover_job(name)
            if job is None:
                if record['submit_intent']:
                    # A crash after sbatch but before its ID was persisted is
                    # ambiguous. Wait for accounting rather than double-submit.
                    for _ in range(10):
                        time.sleep(15)
                        job = self.recover_job(name)
                        if job is not None: break
                    if job is None: raise RuntimeError(f'unresolved submission intent for {name}; reconcile scheduler before retry')
                else:
                    record['submit_intent'] = True; store(ledger_path,jobs)
            if job is None:
                # A partially completed base_train run cannot blindly requeue:
                # the launcher refuses its existing config unless explicitly
                # resumed. Keep all pretraining in the normal partition.
                partition = 'mit_normal_gpu,mit_preemptable' if label.startswith('ft-') else 'mit_normal_gpu'
                requeue = '--requeue' if label.startswith('ft-') else '--no-requeue'
                out = execute(['sbatch', '--parsable', requeue, '-p', partition,
                               f'--nice={self.nice}', f'--job-name={name}', f'--output={directory}/{name}-%j.out', path], capture=True)
                job = out.strip().split(';')[0]
                if not job.isdigit():
                    raise RuntimeError(f'could not parse submitted job ID: {out!r}')
            record['job_id'] = job; store(ledger_path, jobs)
            print(f'submitted/recovered {job} {path}', flush=True)
        ids = {j['job_id'] for j in jobs.values()}
        if not ids:
            return
        self.phase(label, jobs={p:r['job_id'] for p,r in jobs.items()})
        if not wait:
            return
        # Requeued jobs remain in squeue. Only accept top-level completed 0:0
        # accounting after disappearance from the queue; wait for sacct lag.
        missing_since = {}
        active_states = {'PENDING','RUNNING','COMPLETING','CONFIGURING','RESIZING','SUSPENDED','REQUEUED',
                         'REQUEUE_FED','REQUEUE_HOLD','RESV_DEL_HOLD','SIGNALING','STAGE_OUT'}
        while True:
            active = execute(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i'], capture=True).split()
            pending = ids.intersection(active)
            rows = execute(['sacct', '-n', '-X', '-P', '-j', ','.join(sorted(ids)), '--format=JobIDRaw,State,ExitCode'], capture=True)
            result = {r[0]: (r[1], r[2]) for r in (l.split('|') for l in rows.splitlines()) if len(r) >= 3 and r[0] in ids}
            for jid in ids:
                if jid in pending or result.get(jid)==('COMPLETED','0:0'):
                    missing_since.pop(jid,None); continue
                status = result.get(jid,('UNKNOWN',''))[0].split()[0].rstrip('+')
                if status not in active_states | {'UNKNOWN','PREEMPTED'}:
                    raise RuntimeError(f'job {jid} ended {result[jid]}; review {directory}')
                if status=='PREEMPTED' and not label.startswith('ft-'):
                    raise RuntimeError(f'pretraining job {jid} preempted; review {directory}')
                since = missing_since.setdefault(jid,time.monotonic())
                if time.monotonic()-since > 1200:
                    raise RuntimeError(f'job {jid} not in queue and completion unconfirmed for 20 minutes; review accounting')
            if not pending and all(result.get(j) == ('COMPLETED','0:0') for j in ids):
                break
            time.sleep(self.poll)

    def pretrain_jobs(self, directory, stage, extra=(), selection=None, optimizer=None):
        args = ['jobs','--experts',self.e,'--stage',stage,'--out',directory,'--repo-dir',self.repo,'--scratch-base',self.scratch,*extra]
        if selection: args += ['--selection',selection]
        if optimizer: args += ['--optimizer',optimizer]
        self.module('scripts.pretrain_sweep', args)

    def submit_smoke(self, wait=True):
        def smoke_jobs(directory):
            for opt in OPTS:
                (directory/f'e{self.e}-smoke-{opt}-s42.sbatch').write_text(smoke_job(self.e,opt,self.repo,self.scratch))
        self.wave('smoke',smoke_jobs,wait=wait)

    def run(self):
        self.submit_smoke()
        for opt in OPTS:
            name = f'e{self.e}-smoke-{opt}-s42'
            result = load(self.checkpoints/name/'result.json')
            settings = result.get('settings',{})
            expected = dict(optimizer_name=opt,seed=42,n_experts=self.e,train_tokens=262144,
                            total_batch_size=262144,disable_routing_loss=self.e==1,
                            balance_coef=0. if self.e==1 else .01,z_coef=0. if self.e==1 else .001)
            if (result.get('status') != 'completed' or result.get('steps_trained') != 1 or
                    result.get('run') != name or result.get('tokens_trained') != 262144 or
                    not all(isinstance(result.get(k),(int,float)) and math.isfinite(result[k])
                            for k in ('final_val_bpb','final_heldout_bpb','final_train_loss')) or
                    any(settings.get(k)!=v for k,v in expected.items())):
                raise RuntimeError(f'E={self.e} {opt} GPU smoke failed validation')
        extras = {o: set() for o in OPTS}
        needed = set(OPTS)
        for round_ in range(1000):
            extra = edge_args(extras,'--extra-lr')
            self.wave(f'sweep-{round_:03d}', lambda d: self.pretrain_jobs(d,'sweep',extra),
                      filter_scripts=lambda p: any(p.stem.endswith('-'+opt) for opt in needed))
            rc = self.module('scripts.pretrain_sweep',['select','--experts',self.e,'--checkpoints-root',self.checkpoints,'--out',self.selection,*extra],allowed=(0,3))
            sweep_file = self.selection/'lr_selection.json'
            if rc == 0: break
            decisions = load(sweep_file)
            additions = [(o,x['next_lr']) for o,x in decisions.items() if x.get('status') != 'selected']
            if not additions or any(lr in extras[o] for o,lr in additions): raise RuntimeError('sweep extension made no progress')
            for o,lr in additions: extras[o].add(lr)
            needed = {o for o,_ in additions}
        else: raise RuntimeError('sweep extension did not converge')
        factors = {o: set() for o in OPTS}
        for round_ in range(1000):
            extra = edge_args(factors,'--extra-factor')
            # Old result files are validated by transfer-check; do not use GPU
            # jobs merely to skip previously completed transfer candidates.
            self.wave(f'transfer-{round_:03d}',lambda d: self.pretrain_jobs(d,'transfer',extra,sweep_file),
                      filter_scripts=lambda p: not (self.checkpoints/p.stem/'result.json').is_file())
            rc = self.module('scripts.pretrain_sweep',['transfer-check','--experts',self.e,'--selection',sweep_file,'--checkpoints-root',self.checkpoints,'--out',self.selection,*extra],allowed=(0,3))
            final_file = self.selection/'lr_selection_3b.json'
            if rc == 0: break
            report = load(self.selection/'transfer_check.json')
            if any(x['status'] not in ('interior','edge') for x in report.values()): raise RuntimeError('unfittable transfer check; scientific review needed')
            additions = [(o,x['next_factor']) for o,x in report.items() if x['status']=='edge']
            if not additions or any(f in factors[o] for o,f in additions): raise RuntimeError('transfer extension made no progress')
            for o,f in additions: factors[o].add(f)
        else: raise RuntimeError('transfer extension did not converge')
        self.wave('main',lambda d: self.pretrain_jobs(d,'main',selection=final_file),filter_scripts=lambda p: not (self.checkpoints/p.stem/'result.json').is_file())
        self.module('scripts.pretrain_sweep',['base-manifest','--experts',self.e,'--selection',final_file,'--checkpoints-root',self.checkpoints,'--out',self.selection])
        pin = self.selection/'base_manifest.json'
        bases = [f'e{self.e}-main-{o}-s42' for o in OPTS]
        selection = load(final_file)
        def ft_jobs(d, dataset_args, pair_args=()):
            self.module('scripts.finetune_jobs',['--selection',final_file,'--base-manifest',pin,'--base-runs',*bases,
                *dataset_args,'--d0-file',self.scratch/'heldout/fineweb-edu-100b/d0.parquet',
                '--seeds',1,2,3,'--out-root',self.ft_root,'--out',d,'--repo-dir',self.repo,'--job-prefix','ftpilot',*pair_args])
        dataset_arg = lambda name: ['--dataset',f'{name}={self.scratch}/finetune_data/{name}']
        self.wave('ft-initial',lambda d: ft_jobs(d,[v for name in HORIZONS for v in dataset_arg(name)]))
        for round_ in range(1000):
            plan_path = self.work/f'ft-round-{round_:03d}.json'
            # Finish a persisted extension plan before trying to analyze results
            # that may contain only some of that round's paired seed records.
            if plan_path.exists():
                extensions = {tuple(k):fs for k,fs in load(plan_path)}
            else:
                analyses = []
                for dataset,horizons in HORIZONS.items():
                    for steps in horizons:
                        out = self.work/'analysis'/f'{dataset}-{steps}'
                        self.module('scripts.analyze_finetune',['--root',self.ft_root,'--base-manifest',pin,'--bases',*bases,
                            '--dataset',dataset,'--steps',steps,'--seeds',1,2,3,'--out',out])
                        analyses.append((dataset,load(out/'analysis.json')))
                extensions = ft_extensions(analyses,selection)
                store(plan_path,[[list(k),fs] for k,fs in sorted(extensions.items())])
            if not extensions: break
            def extension_jobs(d):
                for (dataset,pre,ft),fs in sorted(extensions.items()):
                    pair = f'{pre}:{ft}'; fracs = ','.join(map(str,fs))
                    ft_jobs(d,dataset_arg(dataset),['--pairs',pair,'--pair-lr-fractions',f'{pair}={fracs}'])
            self.wave(f'ft-edge-{round_:03d}',extension_jobs)
        else: raise RuntimeError('fine-tuning extension did not converge')
        self.phase('complete',jobs={},base_manifest=str(pin),selection=str(final_file),analysis=str(self.work/'analysis'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experts',type=int,choices=(1,4,16),required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--scratch-base',type=Path,required=True)
    parser.add_argument('--poll-seconds',type=int,default=60)
    parser.add_argument('--nice',type=int,default=0)
    parser.add_argument('--initial-submit-only',action='store_true',help='submit smoke jobs and return; the same work directory can be resumed by the CPU coordinator')
    args=parser.parse_args()
    if args.poll_seconds < 10 or args.nice < 0: parser.error('poll interval >=10s and nonnegative nice required')
    args.work_dir.mkdir(parents=True,exist_ok=True)
    with (args.work_dir/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        coordinator=Coordinator(args)
        try:
            if args.initial_submit_only: coordinator.submit_smoke(wait=False)
            else: coordinator.run()
        except Exception as exc:
            coordinator.phase('blocked',error=str(exc))
            raise


if __name__ == '__main__': main()
