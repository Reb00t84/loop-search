#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сборка PDF препринта из md-исходников (v4+). Кладётся в репо (напр. tools/build/),
рядом — header.tex и файлы фигур.

Использование:
    python3 build_preprint.py causal_loop_draft_v4.md          # EN
    python3 build_preprint.py causal_loop_draft_ru_v4.md --ru  # RU (lang=ru, обязательно!)

Что делает:
1. Тайтл-блок (первые строки: '# Заголовок' / '**Автор**' / '*Препринт vN — дата*')
   переносится в YAML-метаданные pandoc -> центрированный тайтл.
2. Фигуры вставляются НАД подписями '**Fig. N.**' / '**Рис. N.**' (маппинг ниже).
3. Перед таблицей §10 вставляется \\needspace (не рвать таблицу у низа страницы).
4. pandoc -f markdown+autolink_bare_uris (голые URL -> \\url, xurl их переносит).
5. xelatex x2, затем QC: overfull=0, missing characters=0, FFFD=0 — иначе exit 1.

Требования: pandoc, xelatex, шрифты DejaVu (+ DejaVu Math TeX Gyre для ℐ/≲/≳),
texlive-lang-cyrillic (русские переносы; проверяется — без них RU-сборка бракуется).
"""
import subprocess, sys, os, re

FIGS_EN = {1:'fig1_penrose_loop.pdf', 2:'fig2_parameter_plane.pdf',
           3:'fig3_archival_funnel.png', 4:'fig4_dv_highres.png'}
FIGS_RU = {1:'fig1_penrose_loop_ru.pdf', 2:'fig2_parameter_plane_ru.pdf',
           3:'fig3_archival_funnel_ru.png', 4:'fig4_dv_highres_ru.png'}
NEEDSPACE_ANCHORS = ['| Target | Survey | z_abs |', '| Цель | Обзор | z_abs |']
FIG_WIDTH = '80%'

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r

def main():
    src = sys.argv[1]
    ru = '--ru' in sys.argv
    tag, figs = ('Рис', FIGS_RU) if ru else ('Fig', FIGS_EN)
    base = os.path.splitext(os.path.basename(src))[0]
    build_md, tex, pdf, log = f'_build_{base}.md', f'_build_{base}.tex', re.sub(r'_ru_v(\d+)$', r'_v\1_ru', base.replace('draft','preprint'))+'.pdf', f'_build_{base}.log'

    t = open(src, encoding='utf-8').read()
    lines = t.split('\n')
    assert lines[0].startswith('# ') and lines[2].startswith('**') and lines[4].startswith('*'), 'тайтл-блок не распознан'
    title, author, date = lines[0][2:].strip(), lines[2].strip('* '), lines[4].strip('* ')
    body = '\n'.join(lines[5:]).lstrip('\n')

    for num, f in figs.items():
        cap = f'**{tag}. {num}.**'
        assert body.count(cap) == 1, f'подпись {cap}: {body.count(cap)} вхождений'
        assert os.path.exists(f), f'нет файла фигуры {f}'
        body = body.replace(cap, f'![]({f}){{width={FIG_WIDTH}}}\n\n{cap}')
    for a in NEEDSPACE_ANCHORS:
        if body.count(a) == 1:
            body = body.replace(a, '\\needspace{18\\baselineskip}\n\n' + a)
    open(build_md, 'w', encoding='utf-8').write(
        f'---\ntitle: "{title}"\nauthor: "{author}"\ndate: "{date}"\n---\n\n' + body)

    cmd = ['pandoc', build_md, '-f', 'markdown+autolink_bare_uris', '-s', '-o', tex,
           '--pdf-engine=xelatex', '-H', 'header.tex',
           '-V', 'mainfont=DejaVu Serif', '-V', 'sansfont=DejaVu Sans',
           '-V', 'monofont=DejaVu Sans Mono', '-V', 'fontsize=10pt',
           '-V', 'geometry:margin=2cm', '-V', 'colorlinks=true']
    if ru:
        cmd += ['-V', 'lang=ru']
    r = run(cmd)
    assert r.returncode == 0, r.stderr
    for _ in range(2):
        run(['xelatex', '-interaction=nonstopmode', tex])
    texlog = open(tex.replace('.tex', '.log'), encoding='utf-8', errors='replace').read()

    # QC
    overfull = texlog.count('Overfull')
    missing = len(re.findall(r'Missing character', texlog))
    nohyph = 'No hyphenation patterns were loaded' in texlog and ru
    txt = run(['pdftotext', tex.replace('.tex', '.pdf'), '-']).stdout
    fffd = txt.count('�')
    os.replace(tex.replace('.tex', '.pdf'), pdf)
    print(f'{pdf}: overfull={overfull} missing_chars={missing} FFFD={fffd} ru_hyph_broken={nohyph}')
    if overfull or missing or fffd or nohyph:
        print('QC FAILED', file=sys.stderr)
        sys.exit(1)
    print('QC OK')

if __name__ == '__main__':
    main()
