#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
continual_eval.py  |  سنجه‌های استاندارد یادگیری پیوسته

بدون هيچ وابستگی بيرونی — فقط کتابخانهٔ استاندارد پايتون.
(روی Termux و سندباکس بدون نصب هيچ بسته‌ای اجرا می‌شود.)

ورودی:
    ماتريس R به‌صورت JSON يا JSONL
    R[i][j] = عملکرد روی تسک j پس از پايان آموزش تسک i
    خانه‌های نامعلوم را null بگذاريد.

خروجی:
    ACC        ميانگين عملکرد نهايی روی همهٔ تسک‌ها
    BWT        انتقال رو به عقب (منفی = فراموشی)
    Forgetting ميانگين افت تسک‌های قديمی (صفر = بدون فراموشی)
    FWT        انتقال رو به جلو
    R          نسبت بازيابی = هزينهٔ ترميم افت / کل بودجهٔ يادگيری

اين فايل فقط اندازه‌گيری است؛ هيچ چيزی از مکانيزم DSA در آن نيست.

نحوهٔ استفاده:
    python continual_eval.py results.json
    python continual_eval.py results.jsonl
    python continual_eval.py results.json --json
    python continual_eval.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    'ContinualMatrix', 'recovery_ratio', 'fade_profile',
    'format_report', 'load_results', 'simulate_scenario',
]

VERSION = '1.1.0'


def _mean(values):
    """ميانگين با ناديده گرفتن مقادير نامعلوم."""
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _fmt(x, nd=4):
    if x is None:
        return '  --  '
    if isinstance(x, float) and not math.isfinite(x):
        return '  --  '
    return str(round(float(x), nd))


# ============================================================================
# 1) ماتريس R
# ============================================================================

class ContinualMatrix:
    """
    ماتريس عملکرد يادگيری پیوسته.

    R[i][j] = عملکرد روی تسک j پس از اتمام آموزش تسک i
    فقط i >= j معنا دارد. خانه‌های ديگر None هستند.

    تسک‌ها به ترتيب زمانی مرتب‌اند: 0، 1، 2، ...
    """

    def __init__(self, R, task_names=None, base_scores=None):
        if not isinstance(R, (list, tuple)) or not R:
            raise ValueError('ماتريس R بايد يک فهرست ناتهی از سطرها باشد.')
        T = len(R)
        grid = []
        for i, row in enumerate(R):
            if not isinstance(row, (list, tuple)):
                raise ValueError('سطر ' + str(i) + ' يک فهرست نيست.')
            if len(row) != T:
                raise ValueError('ماتريس R بايد مربع باشد؛ سطر ' + str(i)
                                 + ' طول ' + str(len(row)) + ' دارد، انتظار ' + str(T))
            grid.append([(None if v is None else float(v)) for v in row])
        self.R = grid
        self.T = T
        self.task_names = list(task_names) if task_names else [
            'تسک ' + str(i) for i in range(T)]
        if len(self.task_names) != T:
            raise ValueError('تعداد نام تسک‌ها با ماتريس نمی‌خواند.')
        if base_scores is None:
            base_scores = [0.0] * T
        self.base_scores = [float(x) for x in base_scores]

    # --- سازنده‌ها ---------------------------------------------------------
    @classmethod
    def from_dict(cls, d):
        return cls(R=d['R'], task_names=d.get('task_names'),
                   base_scores=d.get('base_scores'))

    @classmethod
    def from_jsonl(cls, path):
        """
        هر خط يک رکورد: {"after_task": i, "task": j, "score": x}
        """
        T = 0
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
                T = max(T, int(rec['after_task']) + 1, int(rec['task']) + 1)
        if T == 0:
            raise ValueError('فايل خالی است.')
        R = [[None] * T for _ in range(T)]
        for rec in records:
            R[int(rec['after_task'])][int(rec['task'])] = float(rec['score'])
        return cls(R)

    # --- دسترسي‌ها --------------------------------------------------------
    def diagonal(self):
        """عملکرد هر تسک بلافاصله پس از يادگيری خودش: R[i][i]"""
        return [self.R[i][i] for i in range(self.T)]

    def final_row(self):
        """عملکرد نهايی روی همهٔ تسک‌ها: R[T-1][:]"""
        return list(self.R[self.T - 1])

    # --- سنجه‌ها ----------------------------------------------------------
    def acc(self):
        """ACC = ميانگين عملکرد نهايی روی همهٔ تسک‌ها."""
        return _mean(self.final_row())

    def bwt(self):
        """
        BWT = ميانگين (عملکرد نهايی تسک قديمی - عملکرد همان تسک در لحظهٔ يادگيری)
        منفی = فراموشی؛ نزديک صفر يا مثبت = حفظ دانش.
        """
        if self.T < 2:
            return 0.0
        final, diag = self.final_row(), self.diagonal()
        vals = []
        for j in range(self.T - 1):
            if final[j] is None or diag[j] is None:
                continue
            vals.append(final[j] - diag[j])
        return _mean(vals)

    def forgetting(self):
        """
        Forgetting = ميانگين (بهترين عملکرد گذشتهٔ تسک - عملکرد نهايی همان تسک)
        صفر = هيچ فراموشی‌ای رخ نداده.
        """
        if self.T < 2:
            return 0.0
        final = self.final_row()
        vals = []
        for j in range(self.T - 1):
            past = [self.R[i][j] for i in range(j, self.T)]
            past = [p for p in past if p is not None]
            if not past or final[j] is None:
                continue
            vals.append(max(0.0, max(past) - final[j]))
        return _mean(vals)

    def fwt(self):
        """FWT = ميانگين (عملکرد تسک جديد پيش از آموزش آن - عملکرد پايه)."""
        if self.T < 2:
            return 0.0
        vals = []
        for j in range(1, self.T):
            prev = self.R[j - 1][j]
            if prev is None:
                continue
            vals.append(prev - self.base_scores[j])
        return _mean(vals)

    def retention(self):
        """نسبت حفظ هر تسک قديمی نسبت به لحظهٔ يادگيری."""
        diag, final = self.diagonal(), self.final_row()
        out = []
        for j in range(self.T - 1):
            d = diag[j]
            if d is None or final[j] is None:
                continue
            out.append(1.0 if d <= 0 else float(final[j] / d))
        return out

    def summary(self):
        ret = self.retention()
        return {
            'T': self.T,
            'ACC': self.acc(),
            'BWT': self.bwt(),
            'Forgetting': self.forgetting(),
            'FWT': self.fwt(),
            'final_row': [None if v is None else round(v, 4)
                          for v in self.final_row()],
            'diagonal': [None if v is None else round(v, 4)
                         for v in self.diagonal()],
            'mean_retention': (round(_mean(ret), 4) if ret else None),
            'task_names': self.task_names,
        }


# ============================================================================
# 2) نسبت بازیابی R
# ============================================================================

def recovery_ratio(tokens, scores, target_frac=0.95):
    """
    نسبت بازیابی = (هزينهٔ ترميم افت) / (کل بودجهٔ يادگيری تسک جديد)

    روش (چهار قدم صريح):
        ۱. گودال: نقطه‌ای که عملکرد روی تسک قديمی بيشترين افت را داشته
        ۲. مرجع پيشين: بهترين عملکرد تا پيش از گودال
        ۳. لحظهٔ بازيابی: اولين اندازه‌گيری پس از گودال که به
           target_frac از مرجع پيشين رسيده باشد
        ۴. هزينهٔ ترميم = توکن‌های صرف‌شده از گودال تا لحظهٔ بازيابی

    چرا از گودال و نه از ابتدا:
        اگر از ابتدا بگيريم، اولين اندازه‌گيری هميشه آستانه را رد می‌کند
        و نسبت بی‌معنا می‌شود.

    تفسير:
        ratio < 0.05  ->  محو کوچک (fade)
        ratio >= 0.4  ->  فراموشی واقعی (forget)
    """
    tokens = [float(t) for t in tokens]
    scores = [float(x) for x in scores]
    if len(tokens) != len(scores) or not tokens:
        raise ValueError('طول دو فهرست بايد برابر و ناتهی باشد.')

    trough_idx = min(range(len(scores)), key=lambda i: scores[i])
    trough_score = scores[trough_idx]
    best_prior = max(scores[:trough_idx + 1])

    result = {
        'trough_idx': trough_idx,
        'trough_score': trough_score,
        'best_prior': best_prior,
        'drop_frac': (0.0 if best_prior <= 0
                      else (best_prior - trough_score) / best_prior),
        'tokens_to_learn': tokens[-1],
        'threshold': target_frac * best_prior,
    }

    if trough_idx == 0 or best_prior <= 0:
        result.update({'tokens_to_recover': 0.0, 'ratio': 0.0,
                       'recovered': True, 'note': 'بدون افت قابل توجه'})
        return result

    threshold = target_frac * best_prior
    rec_idx = None
    for i in range(trough_idx + 1, len(scores)):
        if scores[i] >= threshold:
            rec_idx = i
            break

    if rec_idx is None:
        result.update({'tokens_to_recover': None, 'ratio': None,
                       'recovered': False, 'note': 'هرگز به آستانهٔ بازيابی نرسيد'})
        return result

    spent = tokens[rec_idx] - tokens[trough_idx]
    total = tokens[-1]
    result.update({
        'recovery_idx': rec_idx,
        'tokens_to_recover': spent,
        'ratio': (spent / total) if total > 0 else float('inf'),
        'recovered': True,
        'note': 'بازيابی شد',
    })
    return result


def fade_profile(ratio, small=0.05, large=0.4):
    """برچسب کيفی برای نسبت بازیابی."""
    if ratio is None:
        return 'بازيابی نشد (فراموشی کامل)'
    if ratio <= small:
        return 'fade کوچک (محو، نه فراموشی)'
    if ratio >= large:
        return 'forget (فراموشی واقعی)'
    return 'حالت ميانی'


# ============================================================================
# 3) گزارش
# ============================================================================

def format_report(summary):
    """گزارش متنی خوانا از خلاصهٔ سنجه‌ها."""
    L = []
    L.append('=' * 66)
    L.append('گزارش يادگيری پیوسته')
    L.append('=' * 66)
    L.append('تعداد تسک‌ها (T): ' + str(summary['T']))
    L.append('')
    L.append('  ACC        : ' + _fmt(summary['ACC']) + '   ميانگين عملکرد نهايی')
    L.append('  BWT        : ' + _fmt(summary['BWT']) + '   منفی = فراموشی')
    L.append('  Forgetting : ' + _fmt(summary['Forgetting']) + '   صفر = بدون فراموشی')
    L.append('  FWT        : ' + _fmt(summary['FWT']) + '   انتقال رو به جلو')
    if summary.get('mean_retention') is not None:
        L.append('  حفظ ميانگين: ' + _fmt(summary['mean_retention']))
    L.append('')
    L.append('  عملکرد نهايی به تفکيک تسک:')
    for name, val in zip(summary['task_names'], summary['final_row']):
        bar = ''
        if val is not None:
            bar = '#' * int(round(max(0.0, min(1.0, val)) * 28))
        L.append('    ' + str(name).ljust(16) + ' ' + _fmt(val).rjust(7) + '  ' + bar)
    L.append('=' * 66)
    return chr(10).join(L)


def load_results(path):
    """فايل JSON يا JSONL را به ماتريس تبديل می‌کند."""
    if path.endswith('.jsonl'):
        return ContinualMatrix.from_jsonl(path)
    with open(path, 'r', encoding='utf-8') as f:
        return ContinualMatrix.from_dict(json.load(f))


# ============================================================================
# 4) شبيه‌سازی برای آزمون خود سنجه‌ها
# ============================================================================

def simulate_scenario(T=4, mode='forget', seed=0):
    """
    سناریوی ساختگی برای آزمون خود سنجه‌ها.
    هيچ ربطی به عملکرد واقعی هيچ مدلی ندارد.
    """
    rng = random.Random(seed)
    R = [[None] * T for _ in range(T)]
    peak = 0.90
    drop = {'none': 0.01, 'fade': 0.03, 'forget': 0.22}[mode]
    for i in range(T):
        for j in range(i + 1):
            val = peak - drop * (i - j) + rng.gauss(0, 0.004)
            R[i][j] = round(max(0.0, min(1.0, val)), 6)
    return R


# ============================================================================
# 5) خودآزمايی
# ============================================================================

def _selftest():
    names = ['حافظ', 'عبيد', 'طنز', 'جوک']

    print('--- آزمون ۱: بدون فراموشی ---')
    m = ContinualMatrix(simulate_scenario(4, 'none'), names)
    s = m.summary()
    print(format_report(s))
    assert s['Forgetting'] < 0.05, s
    assert s['BWT'] > -0.05, s

    print('--- آزمون ۲: فراموشی فاجعه‌بار ---')
    m2 = ContinualMatrix(simulate_scenario(4, 'forget'), names)
    s2 = m2.summary()
    print(format_report(s2))
    assert s2['Forgetting'] > 0.4, s2
    assert s2['BWT'] < -0.4, s2

    print('--- آزمون ۳: محو تدريجی ---')
    m3 = ContinualMatrix(simulate_scenario(4, 'fade'), names)
    s3 = m3.summary()
    print(format_report(s3))
    assert 0.05 < s3['Forgetting'] < 0.4, s3

    print('--- آزمون ۴: ترتيب درست سنجه‌ها ---')
    assert s2['ACC'] < s3['ACC'] < s['ACC']
    print('  ACC: forget=' + _fmt(s2['ACC']) + ' < fade=' + _fmt(s3['ACC'])
          + ' < none=' + _fmt(s['ACC']))

    print('--- آزمون ۵: نسبت بازيابی ---')
    tokens = [100, 200, 300, 400, 800, 1600, 3200]
    fade_scores = [0.900, 0.888, 0.885, 0.897, 0.901, 0.903, 0.905]
    r_fade = recovery_ratio(tokens, fade_scores)
    print('  fade   -> ratio=' + _fmt(r_fade['ratio']) + '  افت='
          + _fmt(r_fade['drop_frac'], 3) + '  ' + fade_profile(r_fade['ratio']))
    assert r_fade['recovered'] and r_fade['ratio'] <= 0.05, r_fade

    forget_scores = [0.900, 0.700, 0.550, 0.480, 0.600, 0.800, 0.880]
    r_forget = recovery_ratio(tokens, forget_scores)
    print('  forget -> ratio=' + _fmt(r_forget['ratio']) + '  افت='
          + _fmt(r_forget['drop_frac'], 3) + '  ' + fade_profile(r_forget['ratio']))
    assert r_forget['recovered'] and r_forget['ratio'] >= 0.4, r_forget

    assert r_fade['drop_frac'] < r_forget['drop_frac']
    assert r_fade['ratio'] < r_forget['ratio']

    r_flat = recovery_ratio(tokens, [0.90, 0.91, 0.92, 0.92, 0.93, 0.93, 0.94])
    assert r_flat['ratio'] == 0.0 and r_flat['recovered'], r_flat
    print('  flat   -> ratio=0.0  بدون افت')

    print('--- آزمون ۶: بازيابی‌نشدنی ---')
    r_never = recovery_ratio(tokens, [0.9, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert r_never['recovered'] is False and r_never['ratio'] is None
    print('  ' + fade_profile(None))

    print('--- آزمون ۷: ورودی JSON ---')
    import tempfile
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, 'r.jsonl')
    with open(p, 'w', encoding='utf-8') as f:
        for i in range(3):
            for j in range(i + 1):
                f.write(json.dumps({'after_task': i, 'task': j,
                                    'score': 0.9 - 0.1 * (i - j)}) + chr(10))
    m4 = ContinualMatrix.from_jsonl(p)
    assert m4.T == 3 and abs(m4.R[2][0] - 0.7) < 1e-9
    print('  بارگذاری JSONL درست است.')

    print('--- آزمون ۸: خانه‌های نامعلوم ---')
    R = [[0.90, None, None],
         [0.85, 0.88, None],
         [0.83, 0.86, 0.91]]
    m5 = ContinualMatrix(R, ['الف', 'ب', 'ج'])
    s5 = m5.summary()
    assert s5['final_row'] == [0.83, 0.86, 0.91], s5
    # تسک ۰: بهترين گذشته ۰٫۹۰ و نهايی ۰٫۸۳ => افت ۰٫۰۷
    # تسک ۱: بهترين گذشته ۰٫۸۸ و نهايی ۰٫۸۶ => افت ۰٫۰۲
    # ميانگين = ۰٫۰۴۵
    assert abs(s5['Forgetting'] - 0.045) < 1e-9, s5
    assert s5['ACC'] is not None
    print('  Forgetting=' + _fmt(s5['Forgetting']) + '  ACC=' + _fmt(s5['ACC']))

    print('--- آزمون ۹: تشخيص ورودی بد ---')
    for bad in ([1, 2, 3], [[1, 2, 3]], [[1, 2], [3]], []):
        try:
            ContinualMatrix(bad)
            raise AssertionError('بايد خطا می‌داد: ' + str(bad))
        except ValueError:
            pass
    print('  ورودی‌های نامربع رد شدند.')

    print('--- آزمون ۱۰: سازگاری با خروجی poet_continual_sft ---')
    payload = {'task_names': names,
               'R': [[0.9, None, None, None],
                     [0.87, 0.88, None, None],
                     [0.86, 0.87, 0.89, None],
                     [0.85, 0.86, 0.88, 0.9]]}
    m6 = ContinualMatrix.from_dict(payload)
    s6 = m6.summary()
    assert s6['T'] == 4 and 0 < s6['ACC'] <= 1
    print('  خواندن ماتريس خروجی poet درست است. ACC=' + _fmt(s6['ACC']))

    print()
    print('همهٔ ۱۰ آزمون گذشت. نسخهٔ ' + VERSION)


def main(argv=None):
    ap = argparse.ArgumentParser(description='سنجه‌های يادگيری پیوسته')
    ap.add_argument('results', nargs='?', help='فايل JSON يا JSONL')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--json', action='store_true', help='خروجی JSON')
    ap.add_argument('--demo', action='store_true',
                    help='نمونهٔ سه سناريو برای مقايسه')
    args = ap.parse_args(argv)

    if args.demo:
        for mode, label in [('none', 'آموزش مشترک'),
                            ('fade', 'محو تدريجی'),
                            ('forget', 'فراموشی فاجعه‌بار')]:
            m = ContinualMatrix(simulate_scenario(4, mode),
                                ['حافظ', 'عبيد', 'طنز', 'جوک'])
            s = m.summary()
            print(chr(10) + '### سناريو: ' + label)
            print(format_report(s))
        return 0

    if args.selftest or not args.results:
        _selftest()
        return 0

    m = load_results(args.results)
    s = m.summary()
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        print(format_report(s))
    return 0


if __name__ == '__main__':
    sys.exit(main())
