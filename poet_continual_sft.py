#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poet_continual_sft.py  |  اسکلت آموزش پيوسته برای «مغز شاعر»

مسير درست (مهم):
    مدل پايه (HuggingFace / safetensors)
        -> آموزش ترتیبی روی تسک‌ها (SFT + LoRA)
        -> ارزيابی روی همهٔ تسک‌های ديده‌شده  (ساخت ماتريس R)
        -> ادغام آداپتور در وزن‌ها (merge)
        -> تبديل به GGUF
        -> استنتاج

    روي فايل .gguf آموزش نمی‌دهيم؛ GGUF فرمت استنتاج است.

چه چيزی اينجا هست:
    - بارگذاری ديتاست چندمرحله‌ای (JSONL)
    - حلقهٔ آموزش ترتیبی با LoRA
    - ارزيابی پس از هر مرحله روی همهٔ تسک‌ها -> ماتريس R
    - درز خالی برای اتصال مکانيزم خصوصی شما روی لايه‌های توجه

چه چيزی اينجا نيست:
    هيچ بخشی از فرمول/فرکانس/چرخش/نگهبان‌های مکانيزم شما.
    تابع هوک با نام apply_spiral_attenuation يک درز خالی است که شما پر می‌کنيد.

نحوهٔ استفاده:
    python poet_continual_sft.py --config config.json
    python poet_continual_sft.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    'VERSION', 'TaskSpec', 'StageResult', 'DEFAULT_TASKS',
    'read_jsonl', 'write_jsonl', 'load_tasks', 'make_prompt', 'mask_prompt',
    'build_matrix', 'save_matrix', 'score_to_unit',
    'NoOpAttentionHook', 'register_attention_hook',
    'perplexity', 'perplexity_score', 'evaluate_tasks',
    'run_continual', 'merge_adapter', 'export_gguf_command',
]

VERSION = '1.0.0'


# ============================================================================
# 1) تعريف تسک‌ها
# ============================================================================

DEFAULT_TASKS = [
    {'name': 'حافظ', 'file': 'data/hafez.jsonl'},
    {'name': 'عبيد', 'file': 'data/obeyd.jsonl'},
    {'name': 'طنز', 'file': 'data/tanz.jsonl'},
    {'name': 'جوک', 'file': 'data/jok.jsonl'},
]


@dataclass
class TaskSpec:
    name: str
    file: str
    limit: Optional[int] = None
    eval_file: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        return cls(name=d['name'], file=d['file'],
                   limit=d.get('limit'), eval_file=d.get('eval_file'))


@dataclass
class StageResult:
    stage: int
    task: str
    train_seconds: float = 0.0
    train_loss: Optional[float] = None
    tokens_seen: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    adapter_dir: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ============================================================================
# 2) داده
# ============================================================================

def read_jsonl(path, limit=None):
    """
    هر خط: {"instruction": "...", "response": "..."}
    کليدهای پذيرفته‌شده برای متن: instruction/input/prompt و response/output/answer
    """
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, str):
                rec = {'instruction': '', 'response': rec}
            instr = (rec.get('instruction') or rec.get('input')
                     or rec.get('prompt') or '')
            resp = (rec.get('response') or rec.get('output')
                    or rec.get('answer') or '')
            if not str(resp).strip():
                continue
            out.append({'instruction': str(instr), 'response': str(resp)})
            if limit and len(out) >= int(limit):
                break
    return out


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
    return path


def load_tasks(specs):
    """فهرست TaskSpec را به ديتاست واقعی تبديل می‌کند."""
    tasks = []
    for spec in specs:
        spec = spec if isinstance(spec, TaskSpec) else TaskSpec.from_dict(spec)
        train = read_jsonl(spec.file, limit=spec.limit)
        if spec.eval_file and os.path.exists(spec.eval_file):
            ev = read_jsonl(spec.eval_file, limit=spec.limit)
        else:
            ev = train[:max(1, min(32, len(train) // 5))] if train else []
        tasks.append({'name': spec.name, 'file': spec.file,
                      'train': train, 'eval': ev})
    return tasks


# ============================================================================
# 3) قالب پرامپت و ماسک
# ============================================================================

PROMPT_TEMPLATE = (
    '### دستور' + chr(10) + '{instruction}' + chr(10)
    + '### پاسخ' + chr(10)
)


def make_prompt(instruction, response=None):
    text = PROMPT_TEMPLATE.format(instruction=instruction)
    if response is not None:
        text += response
    return text


def mask_prompt(labels, prompt_len):
    """
    ماسک کردن بخش دستور تا مدل فقط روی توليد پاسخ آموزش ببيند.
    labels: فهرست شناسه‌ها؛ prompt_len: تعداد توکن‌های دستور
    """
    return [-100] * int(prompt_len) + list(labels[int(prompt_len):])


# ============================================================================
# 4) درز هوک توجه  (خالی — شما پر می‌کنيد)
# ============================================================================

class NoOpAttentionHook:
    """
    هوک بی‌اثر. مکانيزم شما اين را جایگزين می‌کند.

    امضا: hook(module, args, output, ctx) -> Optional[Any]
        خروجی None  = دست نزن
        خروجی ديگر  = همان مقدار به‌عنوان خروجی لايهٔ توجه استفاده می‌شود
    """
    name = 'noop'

    def __call__(self, module, args, output, ctx):
        return None


def register_attention_hook(model, hook, layer_pattern='self_attn'):
    """
    ثبت هوک روی همهٔ لايه‌های توجه مدل.

    اين فقط سيم‌کشی استاندارد PyTorch است. محتوای مکانيزم شما
    در تابع hook می‌نشيند و هيچ‌کجای اين فايل نيست.

    برمی‌گرداند: فهرست handle‌ها (برای حذف با .remove())
    """
    from functools import partial

    handles = []
    for name, module in model.named_modules():
        if layer_pattern not in name:
            continue
        if not hasattr(module, 'forward'):
            continue
        state = {'name': name}

        def _fn(mod, args, output, _state=state):
            return hook(mod, args, output, _state)

        handles.append(module.register_forward_hook(_fn))
    return handles


def load_attention_hook(spec=None):
    """
    بارگذاری مکانيزم خصوصی در صورت وجود.
    spec می‌تواند «module:callable» يا نام ماژول باشد.
    """
    spec = spec or os.environ.get('POET_ATTENTION_HOOK', '').strip()
    if not spec:
        return NoOpAttentionHook()
    try:
        import importlib
        if ':' in spec:
            mod_name, attr = spec.split(':', 1)
        else:
            mod_name, attr = spec, 'get_hook'
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        return obj() if callable(obj) else obj
    except Exception as e:
        print('[poet] بارگذاری هوک ناموفق بود؛ NoOp فعال شد: ' + str(e))
        return NoOpAttentionHook()


# ============================================================================
# 5) ارزيابی
# ============================================================================

def perplexity(total_nll, total_tokens):
    """سرگشتگی از مجموع NLL و تعداد توکن."""
    if total_tokens <= 0:
        return float('inf')
    return float(math.exp(min(50.0, total_nll / total_tokens)))


def score_to_unit(ppl, lo=1.0, hi=30.0):
    """
    تبديل سرگشتگی به نمرهٔ ۰..۱ (بالاتر بهتر) تا در ماتريس R بنشيند.
    پيش‌فرض: ppl=1 -> 1.0 ، ppl>=30 -> 0.0
    """
    if ppl is None or not math.isfinite(ppl):
        return 0.0
    if ppl <= lo:
        return 1.0
    if ppl >= hi:
        return 0.0
    return float(1.0 - (ppl - lo) / (hi - lo))


def perplexity_score(model, tokenizer, texts, max_length=512, device='cpu'):
    """سرگشتگی مدل روی فهرست متن‌ها (بدون گراديان)."""
    import torch

    model.eval()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors='pt', truncation=True,
                            max_length=max_length)
            ids = enc['input_ids'].to(device)
            if ids.shape[1] < 2:
                continue
            out = model(input_ids=ids, labels=ids)
            n = ids.shape[1] - 1
            total_nll += float(out.loss) * n
            total_tok += n
    return perplexity(total_nll, total_tok)


def evaluate_tasks(model, tokenizer, tasks, upto=None, **kw):
    """
    ارزيابی روی تسک‌های 0..upto و برگرداندن نمرهٔ هر تسک.
    خروجی: dict {نام تسک: نمرهٔ ۰..۱}
    """
    scores = {}
    last = len(tasks) - 1 if upto is None else int(upto)
    for j in range(last + 1):
        task = tasks[j]
        evals = task['eval'] or task['train']
        texts = [make_prompt(r['instruction'], r['response']) for r in evals]
        if not texts:
            scores[task['name']] = 0.0
            continue
        ppl = perplexity_score(model, tokenizer, texts, **kw)
        scores[task['name']] = score_to_unit(ppl)
    return scores


# ============================================================================
# 6) ساخت ماتريس R و ذخيره
# ============================================================================

def build_matrix(task_names, stage_scores):
    """
    stage_scores: فهرست dictها؛ عنصر i ام نمرهٔ تسک‌های 0..i پس از مرحلهٔ i.
    خانه‌های نامعلوم None می‌شوند.
    """
    T = len(task_names)
    R = [[None] * T for _ in range(T)]
    for i, scores in enumerate(stage_scores):
        for j, name in enumerate(task_names):
            if name in scores and j <= i:
                R[i][j] = float(scores[name])
    return R


def save_matrix(path, task_names, R):
    """ذخيره به قالبی که continual_eval.py می‌خواند."""
    clean = [[(None if v is None else float(v)) for v in row] for row in R]
    data = {'task_names': list(task_names), 'R': clean}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


# ============================================================================
# 7) حلقهٔ آموزش پيوسته
# ============================================================================

def _build_model(cfg):
    """بارگذاری مدل پايه + آداپتور LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base = cfg['base_model']
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw = {'trust_remote_code': True, 'device_map': cfg.get('device_map', 'auto')}
    if cfg.get('load_in_4bit'):
        from transformers import BitsAndBytesConfig
        load_kw['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    if cfg.get('load_in_4bit'):
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=int(cfg.get('lora_r', 16)),
        lora_alpha=int(cfg.get('lora_alpha', 32)),
        lora_dropout=float(cfg.get('lora_dropout', 0.05)),
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=cfg.get('target_modules',
                               ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
    )
    model = get_peft_model(model, lora)

    # --- درز مکانيزم شما ------------------------------------------------
    hook = load_attention_hook(cfg.get('attention_hook'))
    handles = register_attention_hook(model, hook, cfg.get('hook_layer_pattern',
                                                          'self_attn'))
    print('[poet] هوک توجه: ' + str(getattr(hook, 'name', hook))
          + '  |  لايه‌های ثبت‌شده: ' + str(len(handles)))
    return model, tok, handles


def _train_one_stage(model, tokenizer, task, cfg, stage):
    """آموزش روی يک تسک."""
    import torch
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments

    rows = task['train']
    max_len = int(cfg.get('max_length', 512))

    class SFT(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            full = make_prompt(r['instruction'], r['response'])
            prompt = make_prompt(r['instruction'])
            enc = tokenizer(full, truncation=True, max_length=max_len)
            plen = len(tokenizer(prompt, truncation=True,
                                 max_length=max_len)['input_ids'])
            ids = enc['input_ids']
            item = {'input_ids': ids, 'attention_mask': enc['attention_mask'],
                    'labels': mask_prompt(ids, min(plen, len(ids) - 1))}
            return item

    out_dir = os.path.join(cfg.get('output_dir', 'out'),
                           'stage' + str(stage) + '_' + task['name'])
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=float(cfg.get('epochs', 2)),
        per_device_train_batch_size=int(cfg.get('batch_size', 1)),
        gradient_accumulation_steps=int(cfg.get('grad_accum', 8)),
        learning_rate=float(cfg.get('lr', 2e-4)),
        warmup_ratio=float(cfg.get('warmup_ratio', 0.03)),
        logging_steps=int(cfg.get('logging_steps', 10)),
        save_strategy='no',
        report_to=[],
        fp16=bool(cfg.get('fp16', False)),
        bf16=bool(cfg.get('bf16', True)),
        optim=cfg.get('optim', 'paged_adamw_8bit'),
    )
    trainer = Trainer(model=model, args=args, train_dataset=SFT())
    t0 = time.time()
    res = trainer.train()
    secs = time.time() - t0

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    n_tok = len(rows) * max_len * int(cfg.get('epochs', 2))
    return {'seconds': secs, 'loss': float(getattr(res, 'training_loss', 0.0) or 0.0),
            'tokens': n_tok, 'adapter_dir': out_dir}


def run_continual(cfg):
    """حلقهٔ اصلی: آموزش ترتیبی + ارزيابی پس از هر مرحله."""
    specs = [TaskSpec.from_dict(t) for t in cfg['tasks']]
    tasks = load_tasks(specs)
    names = [t['name'] for t in tasks]
    for t in tasks:
        print('[poet] تسک ' + t['name'] + ': آموزش=' + str(len(t['train']))
              + ' ارزيابی=' + str(len(t['eval'])))

    model, tok, handles = _build_model(cfg)
    results = []
    stage_scores = []
    try:
        for i, task in enumerate(tasks):
            print(chr(10) + '=== مرحلهٔ ' + str(i) + ' : ' + task['name'] + ' ===')
            info = _train_one_stage(model, tok, task, cfg, i)
            scores = evaluate_tasks(model, tok, tasks, upto=i,
                                    max_length=int(cfg.get('max_length', 512)),
                                    device=str(next(model.parameters()).device))
            stage_scores.append(scores)
            sr = StageResult(stage=i, task=task['name'],
                             train_seconds=info['seconds'], train_loss=info['loss'],
                             tokens_seen=info['tokens'],
                             adapter_dir=info['adapter_dir'], scores=scores)
            results.append(sr.to_dict())
            print('[poet] نمره‌ها پس از مرحلهٔ ' + str(i) + ': '
                  + json.dumps(scores, ensure_ascii=False))
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    R = build_matrix(names, stage_scores)
    mpath = save_matrix(cfg.get('matrix_out', 'results.json'), names, R)
    tpath = os.path.join(cfg.get('output_dir', 'out'), 'stages.json')
    os.makedirs(os.path.dirname(tpath) or '.', exist_ok=True)
    with open(tpath, 'w', encoding='utf-8') as f:
        json.dump({'stages': results, 'task_names': names}, f, ensure_ascii=False, indent=2)

    print(chr(10) + '[poet] ماتريس R ذخيره شد: ' + mpath)
    print('[poet] گام بعد:  python continual_eval.py ' + mpath)
    return mpath


# ============================================================================
# 8) ادغام و تبديل به GGUF
# ============================================================================

def merge_adapter(base_model, adapter_dir, out_dir):
    """ادغام آداپتور در وزن‌های پايه (برای تبديل به GGUF لازم است)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(out_dir)
    return out_dir


def export_gguf_command(merged_dir, out_gguf, quant='q4_k_m',
                        llama_cpp_dir='~/llama.cpp'):
    """
    فرمان تبديل به GGUF را می‌سازد (اجرا نمی‌کند).
    يادآوری: اين گام فقط برای استنتاج است، نه آموزش.
    """
    conv = os.path.join(llama_cpp_dir, 'convert_hf_to_gguf.py')
    quant_bin = os.path.join(llama_cpp_dir, 'build/bin/llama-quantize')
    f16 = out_gguf.replace('.gguf', '.f16.gguf')
    return (
        'python3 ' + conv + ' ' + merged_dir + ' --outfile ' + f16 + ' --outtype f16'
        + ' && ' + quant_bin + ' ' + f16 + ' ' + out_gguf + ' ' + quant.upper()
    )


# ============================================================================
# 9) خودآزمايی (بدون نياز به torch)
# ============================================================================

def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix='poet_test_')

    # ۱) خواندن JSONL
    p = os.path.join(tmp, 'hafez.jsonl')
    write_jsonl(p, [
        {'instruction': 'يک بيت حافظ بگو', 'response': 'دوش ديدم که ملائک در ميخانه زدند'},
        {'instruction': 'بيت دوم', 'response': 'گل آدم بسرشتند و به پيمانه زدند'},
    ])
    rows = read_jsonl(p)
    assert len(rows) == 2 and rows[0]['response'].startswith('دوش'), rows

    # ۲) قالب پرامپت
    t = make_prompt('سلام', 'عليک')
    assert t.endswith('عليک') and '### دستور' in t

    # ۳) ماسک
    assert mask_prompt([1, 2, 3, 4], 2) == [-100, -100, 3, 4]

    # ۴) ماتريس R با خانه‌های نامعلوم
    names = ['حافظ', 'عبيد', 'طنز']
    stages = [
        {'حافظ': 0.90},
        {'حافظ': 0.85, 'عبيد': 0.88},
        {'حافظ': 0.83, 'عبيد': 0.86, 'طنز': 0.91},
    ]
    R = build_matrix(names, stages)
    assert R[0][1] is None and R[1][2] is None, R
    assert R[2][0] == 0.83 and R[2][2] == 0.91

    # ۵) نمره‌سازی
    assert score_to_unit(1.0) == 1.0 and score_to_unit(30.0) == 0.0
    assert 0.0 < score_to_unit(15.0) < 1.0
    assert score_to_unit(float('inf')) == 0.0

    # ۶) سرگشتگی
    assert abs(perplexity(0.0, 10) - 1.0) < 1e-9
    assert perplexity(10, 0) == float('inf')

    # ۷) ذخيرهٔ ماتريس و خواندن با continual_eval
    mp = save_matrix(os.path.join(tmp, 'results.json'), names, R)
    with open(mp, encoding='utf-8') as f:
        loaded = json.load(f)
    assert loaded['task_names'] == names and loaded['R'][2][0] == 0.83

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from continual_eval import ContinualMatrix
        m = ContinualMatrix.from_dict(loaded)
        s = m.summary()
        assert s['T'] == 3 and s['Forgetting'] >= 0.0
        print('  پل به continual_eval درست کار می‌کند: ACC='
              + str(round(s['ACC'], 4)) + ' Forgetting='
              + str(round(s['Forgetting'], 4)))
    except ImportError:
        print('  (continual_eval.py کنار اين فايل نيست؛ پل آزمون نشد)')

    # ۸) هوک خالی
    h = load_attention_hook(None)
    assert h.name == 'noop'
    assert h(None, (), None, {}) is None

    # ۹) فرمان GGUF
    cmd = export_gguf_command('/m/merged', '/o/model.gguf')
    assert 'convert_hf_to_gguf.py' in cmd and 'llama-quantize' in cmd
    assert 'Q4_K_M' in cmd

    # ۱۰) تسک پيش‌فرض
    assert len(DEFAULT_TASKS) == 4 and DEFAULT_TASKS[0]['name'] == 'حافظ'

    print('همهٔ ۱۰ آزمون گذشت. نسخهٔ ' + VERSION)


def main(argv=None):
    ap = argparse.ArgumentParser(description='آموزش پيوسته مغز شاعر')
    ap.add_argument('--config', help='فايل JSON تنظيمات')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--merge', nargs=3, metavar=('BASE', 'ADAPTER', 'OUT'))
    ap.add_argument('--gguf-command', nargs=2, metavar=('MERGED_DIR', 'OUT_GGUF'))
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if args.gguf_command:
        print(export_gguf_command(args.gguf_command[0], args.gguf_command[1]))
        return 0
    if args.merge:
        print('ادغام شد در: ' + merge_adapter(*args.merge))
        return 0
    if not args.config:
        ap.print_help()
        return 1
    with open(args.config, encoding='utf-8') as f:
        cfg = json.load(f)
    run_continual(cfg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
