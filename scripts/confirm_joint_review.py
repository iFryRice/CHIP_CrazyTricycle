"""Queue remaining concept checks and confirm the frozen joint pipeline."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from patientphex.data import digest_file,read_jsonl,write_json,write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from scripts.optimize_cpu_association import bootstrap_delta,error_counts,validate_file
from scripts.run_concept_review import prepare as prepare_concepts
from scripts.run_joint_review import inputs
from scripts.run_llm_association import atomic_json,load_plan


def confirm(plan_path,plan):
    report=ROOT/plan['report_dir']
    if (report/'summary.json').exists():raise FileExistsError('Preserve prior joint confirmation.')
    pilot=load_plan(ROOT/plan['parent_plan'])
    parent_summary=json.loads((ROOT/plan['parent_summary']).read_text())
    if not parent_summary['promoted'] or plan['fixed']!=pilot['fixed'] or plan['ensemble_plan']!=pilot['ensemble_plan']:
        raise ValueError('The passing joint pilot or its fixed policy changed.')
    docs,baseline,folds=inputs(plan)
    old_pilot={str(d['pmc_id']):d for d in read_jsonl(ROOT/plan['parent_predictions'])}
    predictions,rows,evidence=[],[],[]
    for fold,ids in enumerate(folds):
        is_pilot=fold in pilot['pilot_folds']
        work=ROOT/(pilot['work_dir'] if is_pilot else plan['work_dir'])/f'fold{fold}'
        summary=json.loads((work/'summary.json').read_text())
        expected=digest_file(ROOT/plan['parent_plan']) if is_pilot else digest_file(plan_path)
        if (summary['status']!='completed' or summary['plan_sha256']!=expected
                or summary['validation_document_ids']!=ids
                or summary['predictions_sha256']!=digest_file(work/'predictions.jsonl')):
            raise ValueError('A joint prediction is incomplete, changed or from another fold.')
        local=read_jsonl(work/'predictions.jsonl')
        if [str(d['pmc_id']) for d in local]!=ids:raise ValueError('Joint fold coverage or ordering changed.')
        if is_pilot and local!=[old_pilot[i] for i in ids]:raise ValueError('Joint pilot no longer replays exactly.')
        validation=[docs[i] for i in ids];base=[baseline[i] for i in ids]
        old,metrics=evaluate(validation,base),evaluate(validation,local)
        row={'fold':fold,'baseline':old,'metrics':metrics,'delta':metrics['score']-old['score']}
        rows.append(row);predictions.extend(local);evidence.append(summary)
        write_json(report/f'fold{fold}_metrics.json',row);write_json(report/f'fold{fold}_inference.json',summary)
        print(json.dumps({'fold':fold,'score':metrics['score'],'gain':row['delta']}),flush=True)
    source,original=list(docs.values()),list(baseline.values())
    old,metrics=evaluate(source,original),evaluate(source,predictions)
    remaining=sum(rows[f]['delta'] for f in plan['pilot_folds'])/len(plan['pilot_folds'])
    bootstrap=bootstrap_delta(source,original,predictions)
    gates=plan['confirmation_requirement']
    checks={'remaining_mean_gain':remaining>=gates['minimum_remaining_mean_gain'],
            'full_score_gain':metrics['score']-old['score']>=gates['minimum_full_score_gain'],
            'nonworse_folds':sum(row['delta']>=0 for row in rows)>=gates['minimum_nonworse_folds'],
            'component_guard':all(metrics[k]['f1']>=old[k]['f1']-gates['maximum_component_f1_loss'] for k in ['mention','document','association_micro','association_macro']),
            'mention_improved':metrics['mention']['f1']>old['mention']['f1'],
            'format_guard':all(item['concept_review']['format_success_rate']>=.98 for item in evidence),
            'positive_bootstrap_lower_bound':bootstrap['interval_95'][0]>0}
    write_jsonl(report/'development_oof.jsonl',predictions)
    validated=validate_file(report/'development_oof.jsonl',source,Ontology(ROOT/plan['ontology_path']),report/'validation.json')
    result={'plan_sha256':digest_file(plan_path),'selected':'joint_all_reviewed','promoted':all(checks.values()),
            'baseline':old,'metrics':metrics,'fold_deltas':[row['delta'] for row in rows],
            'confirmation_mean_gain':remaining,'checks':checks,'bootstrap':bootstrap,'validation':validated,
            'errors':error_counts(source,predictions),'scope':'Fivefold development with the joint policy frozen after two-fold screening. All documents were used by prior development; not independent-test or official B accuracy.'}
    write_json(report/'summary.json',result)
    return result


def archive(plan_path,plan,state):
    report=ROOT/plan['report_dir'];concept_plan=json.loads((ROOT/plan['concept_confirmation_plan']).read_text())
    for fold in plan['pilot_folds']:
        for name in ['prepared.json','preflight.json','summary.json','progress.json']:
            source=ROOT/concept_plan['work_dir']/f'fold{fold}'/name
            if source.exists():shutil.copyfile(source,report/f'fold{fold}_concept_{name}')
    lines=['# 联合流水线全五折确认','',f"计划 SHA256：`{digest_file(plan_path)}`。",'',f"状态：`{state['status']}`。",'',
           '固定使用原 Qwen 概念含义核对规则和三种子关联融合，逐折重算过滤后发生变化的结构及文本输入。没有重新训练、挑选种子或改变阈值。','']
    if (report/'summary.json').exists():
        result=json.loads((report/'summary.json').read_text())
        lines.extend([f"全五折：{result['baseline']['score']:.6f} -> {result['metrics']['score']:.6f}；通过全部门槛：{result['promoted']}。",'',
                      f"逐折增益：`{result['fold_deltas']}`；后三折平均增益：{result['confirmation_mean_gain']:+.6f}。",'',
                      f"检查：`{result['checks']}`；bootstrap 区间：`{result['bootstrap']['interval_95']}`。"])
    if 'error' in state:lines.extend(['',f"错误：`{state['error']}`。"])
    lines.extend(['','门槛在两折试验前声明，沿用后三折平均增益至少 0.003、全五折增益至少 0.005、至少四折不下降、各项 F1 损失不超过 0.002，以及正的 bootstrap 下界。','',
                  '本次仍为反复使用开发文献的结果，既不是独立测试，也不是官方 B 分数。只有通过全部检查才生成新的 B 候选。',''])
    (report/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(report/'artifact_manifest.json',{p.relative_to(report).as_posix():digest_file(p) for p in sorted(report.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})


def queue(plan_path,plan):
    pilot=load_plan(ROOT/plan['parent_plan'])
    result=json.loads((ROOT/plan['parent_summary']).read_text())
    if not result['promoted'] or plan['fixed']!=pilot['fixed'] or plan['confirmation_requirement']!=pilot['confirmation_requirement']:
        raise ValueError('Joint confirmation changed the selected policy or original gates.')
    concept_path=ROOT/plan['concept_confirmation_plan'];concept=load_plan(concept_path)
    original_concept=load_plan(ROOT/pilot['concept_plans']['0'])
    for key in ['protocol_module','generation','fixed']:
        if concept[key]!=original_concept[key]:raise ValueError('Remaining-fold concept inference changed the protocol.')
    report=ROOT/plan['report_dir'];state_path=report/'queue.json'
    if state_path.exists():raise FileExistsError('Inspect existing joint confirmation before restarting.')
    report.mkdir(parents=True,exist_ok=True);write_json(report/'plan.json',plan)
    started=time.time();state={'status':'preparing','pid':os.getpid(),'plan_sha256':digest_file(plan_path),'jobs':[],'started_at_epoch':started}
    pending,active,failed=list(plan['pilot_folds']),{},False
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}

    def launch(fold,gpu,phase):
        wrapper='scripts/run_concept_review.sh' if phase=='concept' else 'scripts/run_joint_review.sh'
        selected_plan=concept_path if phase=='concept' else plan_path
        log=report/f'fold{fold}_{phase}.log'
        with log.open('x',encoding='utf-8') as handle:
            process=subprocess.Popen(['bash',wrapper,'--plan',str(selected_plan),'--fold',str(fold)],cwd=ROOT,
                        env={**env,'LLM_GPU_INDEX':str(gpu)},stdout=handle,stderr=subprocess.STDOUT)
        job={'fold':fold,'gpu':gpu,'phase':phase,'pid':process.pid,'status':'running','log':str(log.relative_to(ROOT))}
        state['jobs'].append(job);active[gpu]=process,job

    try:
        atomic_json(state_path,state)
        prepare_concepts(concept_path,concept)
        while pending or active:
            for gpu in [2,0]:
                if gpu not in active and pending and not failed:launch(pending.pop(0),gpu,'concept')
            state['status']='inference_running';state['pending_folds']=len(pending)
            for gpu,(process,job) in list(active.items()):
                code=process.poll()
                if code is not None:
                    job.update(status='completed' if code==0 else 'failed',exit_code=code)
                    failed|=code!=0;del active[gpu]
                    if code==0 and job['phase']=='concept' and not failed:launch(job['fold'],gpu,'joint')
            atomic_json(state_path,state)
            if failed and not active:raise RuntimeError('Joint confirmation phase failed; pending work was not launched.')
            if pending or active:time.sleep(10)
        state['status']='evaluating';atomic_json(state_path,state)
        result=confirm(plan_path,plan);state.update(status='completed',promoted=result['promoted'],score=result['metrics']['score'])
    except Exception as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        state['elapsed_seconds']=time.time()-started;atomic_json(state_path,state);archive(plan_path,plan,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['queue','evaluate']);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=load_plan(args.plan)
    if args.stage=='queue':queue(args.plan.resolve(),plan)
    else:confirm(args.plan.resolve(),plan)
