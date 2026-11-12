#!/usr/bin/env python3
"""
Odessa Mae — generador de videos para YouTube.

Convierte MP3 + fotos fijas en:
  * sencillos 1920x1080 con cartel inicial y letra quemada (+ .srt),
  * Shorts 1080x1920 con el primer estribillo,
  * una compilación con cartel por canción y timestamps de capítulos.

Uso (desde esta carpeta):
  python3 odessa.py align        [--only 1,2]          transcribe + alinea + escribe .srt
  python3 odessa.py preview      [--song 1] [--seconds 43]
  python3 odessa.py singles      [--only 1,2] [--use-srt]
  python3 odessa.py shorts       [--only 1,2] [--use-srt]
  python3 odessa.py compilation
  python3 odessa.py all          [--use-srt]
  python3 odessa.py verify
  python3 odessa.py report       escribe output/REPORTE.md

--use-srt vuelve a renderizar con los tiempos (y textos) de output/NN-base.srt,
por ejemplo después de corregirlo a mano. En ese modo el .srt no se sobrescribe.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["_songs"] = {s["n"]: s for s in cfg["songs"]}
    return cfg


def P(cfg, key) -> Path:
    return ROOT / cfg["paths"][key]


def song_paths(cfg, s) -> dict:
    nn = f"{s['n']:02d}"
    stem = f"{nn}-{s['base']}"
    work = P(cfg, "work_dir")
    out = P(cfg, "output_dir")
    return {
        "nn": nn,
        "stem": stem,
        "mp3": P(cfg, "mp3_dir") / f"{stem}.mp3",
        "image": P(cfg, "images_dir") / f"foto-{nn}.png",
        "whisper": work / "whisper" / f"{stem}.json",
        "align": work / "align" / f"{stem}.json",
        "norm_wav": work / "audio" / f"{stem}.norm.wav",
        "norm_json": work / "audio" / f"{stem}.loudnorm.json",
        "ass": work / "ass" / f"{stem}.ass",
        "short_ass": work / "ass" / f"short-{nn}.ass",
        "mp4": out / f"{stem}.mp4",
        "srt": out / f"{stem}.srt",
        "short": out / "shorts" / f"short-{nn}.mp4",
    }


# ---------------------------------------------------------------------------
# Utilidades ffmpeg
# ---------------------------------------------------------------------------
def run(cmd: list[str], *, quiet=True, cwd=ROOT) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-4000:])
        raise RuntimeError(f"Falló: {' '.join(cmd[:6])} …")
    return res


def ffprobe_duration(path: Path) -> float:
    res = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)])
    return float(res.stdout.strip())


def audio_samples_duration(path: Path) -> float:
    """Duración exacta de un WAV leyendo el número de muestras."""
    res = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-count_packets",
               "-show_entries", "stream=duration,sample_rate,duration_ts", "-of", "json", str(path)])
    st = json.loads(res.stdout)["streams"][0]
    return float(st["duration"])


def rel(p: Path) -> str:
    """Ruta relativa a ROOT (los filtros de ffmpeg se llevan mal con rutas raras)."""
    return str(Path(p).resolve().relative_to(ROOT))


def ass_color(hex_rgb: str, opacity: float = 1.0) -> str:
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    a = round((1.0 - opacity) * 255)
    return f"&H{a:02X}{b}{g}{r}".upper()


def fmt_ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def fmt_srt_time(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt_time(s: str) -> float:
    h, m, rest = s.strip().replace(".", ",").split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


# ---------------------------------------------------------------------------
# Letras
# ---------------------------------------------------------------------------
@dataclass
class LyricLine:
    text: str
    section: str
    stanza: int


def _parse_lyric_block(block: str) -> list[LyricLine]:
    lines: list[LyricLine] = []
    section, stanza, prev_blank = "", 0, True
    for raw in block.splitlines():
        s = raw.strip()
        if not s:
            if not prev_blank:
                stanza += 1
            prev_blank = True
            continue
        m = re.fullmatch(r"\[(.+?)\]", s)
        if m:
            section = m.group(1).split("|")[0].strip()
            if not prev_blank:
                stanza += 1
            prev_blank = True
            if section.lower() == "end":
                break
            continue
        prev_blank = False
        lines.append(LyricLine(s, section, stanza))
    return lines


def parse_lyrics_md(path: Path) -> dict[int, list[LyricLine]]:
    """Devuelve {número de capítulo: [líneas]}.

    Formatos aceptados dentro de cada sección con encabezado "N. Título":
      A) **Lyrics** seguido de un bloque ```text … ``` (con etiquetas [Verse], [Chorus]…)
      B) una línea LYRICS y otra CREDITS (descripción de YouTube); se toma lo de en medio.
    """
    text = path.read_text(encoding="utf-8")
    heads = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, re.M))
    songs: dict[int, list[LyricLine]] = {}
    for i, h in enumerate(heads):
        body = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        m = re.match(r"(?:Cap(?:ítulo|itulo)?\.?\s*|Chapter\s*)?(\d+)\s*[.:)\-–—]", h.group(2), re.I)
        num = int(m.group(1)) if m else None
        if num is None:
            mc = re.search(r"\bChapter\s+(\d+)\b", body)
            num = int(mc.group(1)) if mc else None
        if num is None:
            continue
        block = None
        ma = re.search(r"\*\*\s*Lyrics\s*\*\*[^\n]*\n+```[a-z]*\n(.*?)```", body, re.S | re.I)
        if ma:
            block = ma.group(1)
        else:
            mb = re.search(r"^\s*LYRICS\s*$\n(.*?)^\s*CREDITS\s*$", body, re.S | re.M)
            if mb:
                block = mb.group(1)
        if block:
            songs[num] = _parse_lyric_block(block)
    return songs


def display_text(line: str, strip_punct: bool) -> str:
    t = line.strip()
    if strip_punct:
        if t.endswith((',"', ",”", '."', ".”")) and not t.endswith('..."'):
            t = t[:-2] + t[-1]
        elif t.endswith((",", ";")) or (t.endswith(".") and not t.endswith("...")):
            t = t[:-1]
    return t


def italic_ass(t: str) -> str:
    return re.sub(r"\(([^)]*)\)", r"{\\i1}(\1){\\i0}", t)


def italic_srt(t: str) -> str:
    return re.sub(r"\(([^)]*)\)", r"<i>(\1)</i>", t)


def srt_to_ass_text(t: str) -> str:
    if "<i>" in t.lower():
        t = re.sub(r"<i>", r"{\\i1}", t, flags=re.I)
        t = re.sub(r"</i>", r"{\\i0}", t, flags=re.I)
        return re.sub(r"<[^>]+>", "", t).replace("\n", r"\N")
    return italic_ass(re.sub(r"<[^>]+>", "", t)).replace("\n", r"\N")


# ---------------------------------------------------------------------------
# Normalización de palabras para la alineación
# ---------------------------------------------------------------------------
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if 1100 <= n <= 9999 and n % 100 != 0 or 1100 <= n < 2000:
        return num_words(n // 100) + " " + (num_words(n % 100) if n % 100 else "hundred")
    return str(n)


def norm_tokens(text: str) -> list[str]:
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"(\d)[:.](\d)", r"\1 \2", t)
    t = t.replace("'", "")
    t = re.sub(r"\d+", lambda m: " " + num_words(int(m.group())) + " ", t)
    t = re.sub(r"[^a-z]+", " ", t)
    return t.split()


# ---------------------------------------------------------------------------
# Transcripción (faster-whisper)
# ---------------------------------------------------------------------------
_whisper_model = None


def load_audio_16k(path: Path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def transcribe(cfg, s, force=False) -> dict:
    sp = song_paths(cfg, s)
    if sp["whisper"].exists() and not force:
        return json.loads(sp["whisper"].read_text())
    global _whisper_model
    from faster_whisper import WhisperModel
    lc = cfg["lyrics"]
    if _whisper_model is None:
        print(f"  cargando modelo {lc['whisper_model']} …", flush=True)
        _whisper_model = WhisperModel(lc["whisper_model"], device="cpu",
                                      compute_type=lc["whisper_compute"],
                                      cpu_threads=lc["whisper_threads"])
    t0 = time.time()
    segs, info = _whisper_model.transcribe(load_audio_16k(sp["mp3"]), language="en",
                                           word_timestamps=True, beam_size=5,
                                           condition_on_previous_text=False, vad_filter=False)
    data = [{"start": x.start, "end": x.end, "text": x.text,
             "words": [{"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}
                       for w in (x.words or [])]} for x in segs]
    out = {"duration": info.duration, "segments": data}
    sp["whisper"].parent.mkdir(parents=True, exist_ok=True)
    sp["whisper"].write_text(json.dumps(out, indent=1))
    print(f"  transcripto en {time.time() - t0:.0f} s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Alineación letra ↔ transcripción
# ---------------------------------------------------------------------------
@dataclass
class TimedLine:
    idx: int
    text: str            # texto de la letra (lo que se muestra)
    section: str
    stanza: int
    start: float = 0.0   # comienzo cantado
    end: float = 0.0     # fin cantado (última palabra)
    located: bool = False
    matched: int = 0
    words: int = 0
    show_start: float = 0.0
    show_end: float = 0.0


def align_lyrics(lines: list[LyricLine], whisper: dict, first_word_max: float = 1.0,
                 first_word_typical: float = 0.6) -> list[TimedLine]:
    from rapidfuzz.distance import Indel
    import numpy as np

    # palabras de la letra: (línea, token, opcional) — lo que está entre paréntesis es
    # la respuesta del coro y puede no aparecer en la transcripción.
    lw: list[tuple[int, str, bool]] = []
    for li, ln in enumerate(lines):
        for part, inside in _split_parens(ln.text):
            for tok in norm_tokens(part):
                lw.append((li, tok, inside))
    tw: list[tuple[str, float, float]] = []
    for seg in whisper["segments"]:
        for w in seg["words"]:
            for tok in norm_tokens(w["word"]):
                tw.append((tok, w["start"], w["end"]))
    n, m = len(lw), len(tw)
    sim = np.zeros((n, m))
    for i, (_, a, _) in enumerate(lw):
        for j, (b, _, _) in enumerate(tw):
            sim[i, j] = Indel.normalized_similarity(a, b)

    def msc(s):
        return 2.0 * s - 0.9 if s >= 0.6 else -1.5
    GAP_T = -0.35
    gap_l = [(-0.05 if opt else -0.6) for (_, _, opt) in lw]

    D = np.zeros((n + 1, m + 1))
    B = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 diag, 2 up (gap en T), 3 left (gap en L)
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + gap_l[i - 1]
        B[i, 0] = 2
    B[0, 1:] = 3  # huecos iniciales de la transcripción: gratis
    for i in range(1, n + 1):
        gl = gap_l[i - 1]
        last_row = (i == n)
        for j in range(1, m + 1):
            d = D[i - 1, j - 1] + msc(sim[i - 1, j - 1])
            u = D[i - 1, j] + gl
            l = D[i, j - 1] + (0.0 if last_row else GAP_T)
            if d >= u and d >= l:
                D[i, j], B[i, j] = d, 1
            elif u >= l:
                D[i, j], B[i, j] = u, 2
            else:
                D[i, j], B[i, j] = l, 3
    # traza
    pairs = {}
    i, j = n, m
    while i > 0 and j > 0:
        b = B[i, j]
        if b == 1:
            if sim[i - 1, j - 1] >= 0.67:
                pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif b == 2:
            i -= 1
        else:
            j -= 1

    timed = [TimedLine(k, ln.text, ln.section, ln.stanza) for k, ln in enumerate(lines)]
    per_line: dict[int, list[tuple[int, int, bool]]] = {}
    for wi, (li, _, opt) in enumerate(lw):
        per_line.setdefault(li, []).append((wi, pairs.get(wi, -1), opt))
    for li, items in per_line.items():
        tl = timed[li]
        req = [x for x in items if not x[2]] or items
        tl.words = len(req)
        tl.matched = sum(1 for x in req if x[1] >= 0)
        good = [x for x in items if x[1] >= 0]
        best = max((sim[x[0], x[1]] for x in good), default=0)
        need = max(1, math.ceil(0.34 * len(req)))
        if tl.matched >= need and best >= 0.8:
            tl.located = True
            first_req = next((k for k, x in enumerate(req) if x[1] >= 0), 0)
            first_tw = next(x[1] for x in req if x[1] >= 0)
            w_start, w_end = tw[first_tw][1], tw[first_tw][2]
            # Whisper suele "estirar" hacia atrás la primera palabra después de una pausa
            # (le suma el silencio previo). Su final es confiable: si dura demasiado,
            # el comienzo real se estima desde el final.
            if w_end - w_start > first_word_max:
                w_start = max(w_start, w_end - first_word_typical)
            tl.start = w_start - 0.3 * first_req
            last_any = max(x[1] for x in good)
            trailing = sum(1 for x in reversed(req) if x[1] < 0) if req[-1][1] < 0 else 0
            tl.end = tw[last_any][2] + 0.25 * trailing
    _fix_order_and_interpolate(timed, whisper.get("duration") or 0)
    return timed


def _split_parens(text: str):
    pos = 0
    for mm in re.finditer(r"\(([^)]*)\)", text):
        if mm.start() > pos:
            yield text[pos:mm.start()], False
        yield mm.group(1), True
        pos = mm.end()
    if pos < len(text):
        yield text[pos:], False


def _fix_order_and_interpolate(timed: list[TimedLine], duration: float):
    # descarta ubicaciones fuera de orden (una línea no puede empezar antes que la anterior)
    last = -1.0
    for tl in timed:
        if tl.located:
            if tl.start < last - 0.05:
                tl.located = False
            else:
                tl.start = max(tl.start, last if last > 0 else 0.0)
                tl.end = max(tl.end, tl.start + 0.3)
                last = tl.start
    # interpola las no ubicadas entre sus vecinas, repartiendo por cantidad de palabras
    k = 0
    while k < len(timed):
        if timed[k].located:
            k += 1
            continue
        a = k - 1
        b = k
        while b < len(timed) and not timed[b].located:
            b += 1
        t0 = timed[a].end if a >= 0 else max(0.0, (timed[b].start if b < len(timed) else duration) - 3.0 * (b - k))
        t1 = timed[b].start if b < len(timed) else min(duration, t0 + 3.0 * (b - k))
        if t1 <= t0:
            t1 = t0 + 0.5 * (b - k)
        weights = [max(1, timed[x].words) for x in range(k, b)]
        tot = sum(weights)
        acc = t0
        for x, w in zip(range(k, b), weights):
            span = (t1 - t0) * w / tot
            timed[x].start = acc
            timed[x].end = acc + span * 0.9
            acc += span
        k = b


def compute_display(timed: list[TimedLine], cfg, duration: float):
    lc = cfg["lyrics"]
    for i, tl in enumerate(timed):
        tl.show_start = max(0.0, tl.start - lc["lead"])
    for i, tl in enumerate(timed):
        nxt = timed[i + 1].show_start if i + 1 < len(timed) else duration - 0.05
        tl.show_end = min(nxt, tl.end + lc["silence_hold"])
        if tl.show_end - tl.show_start < lc["min_display"]:
            tl.show_end = min(nxt, tl.show_start + lc["min_display"])
        tl.show_end = min(tl.show_end, duration - 0.05)


def align_song(cfg, s, lyrics: dict, force=False) -> list[TimedLine]:
    sp = song_paths(cfg, s)
    if s["n"] not in lyrics:
        raise SystemExit(f"No encontré la letra del capítulo {s['n']} en {cfg['paths']['lyrics_file']}")
    whisper = transcribe(cfg, s)
    lc = cfg["lyrics"]
    timed = align_lyrics(lyrics[s["n"]], whisper, lc.get("first_word_max", 1.0), lc.get("first_word_typical", 0.6))
    duration = ffprobe_duration(sp["mp3"])
    compute_display(timed, cfg, duration)
    sp["align"].parent.mkdir(parents=True, exist_ok=True)
    sp["align"].write_text(json.dumps([tl.__dict__ for tl in timed], indent=1, ensure_ascii=False))
    return timed


def load_alignment(cfg, s) -> list[TimedLine]:
    sp = song_paths(cfg, s)
    return [TimedLine(**d) for d in json.loads(sp["align"].read_text())]


def write_srt(path: Path, cues: list[tuple[float, float, str]]):
    out = []
    for k, (a, b, t) in enumerate(cues, 1):
        out.append(f"{k}\n{fmt_srt_time(a)} --> {fmt_srt_time(b)}\n{t}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def read_srt(path: Path) -> list[tuple[float, float, str]]:
    cues = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
        ls = [x for x in block.splitlines() if x.strip()]
        k = next((i for i, x in enumerate(ls) if "-->" in x), None)
        if k is None:
            continue
        a, b = [parse_srt_time(x) for x in ls[k].split("-->")]
        cues.append((a, b, "\n".join(ls[k + 1:])))
    return cues


def lyric_cues(cfg, s, timed: list[TimedLine], use_srt: bool):
    """Lista de (inicio, fin, texto_ass, texto_srt) para quemar en el video."""
    sp = song_paths(cfg, s)
    strip = cfg["lyrics"]["strip_line_end_punct"]
    if use_srt:
        if not sp["srt"].exists():
            raise SystemExit(f"--use-srt: no existe {rel(sp['srt'])}")
        return [(a, b, srt_to_ass_text(t), t) for a, b, t in read_srt(sp["srt"])]
    cues = []
    for tl in timed:
        t = display_text(tl.text, strip)
        cues.append((tl.show_start, tl.show_end, italic_ass(t), italic_srt(t)))
    return cues


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def loudnorm_two_pass(cfg, src_args: list[str], out_wav: Path, out_json: Path, force=False,
                      margin: float | None = None) -> dict:
    if out_wav.exists() and out_json.exists() and not force:
        return json.loads(out_json.read_text())
    ac = cfg["audio"]
    base = f"loudnorm=I={ac['lufs']}:TP={ac['true_peak']}:LRA={ac['lra']}"
    r1 = run(["ffmpeg", "-hide_banner", "-nostats", *src_args, "-af", base + ":print_format=json",
              "-f", "null", "-"])
    m1 = json.loads(re.findall(r"\{[^{}]*\}", r1.stderr)[-1])
    # Modo lineal = solo ganancia. Si para llegar al objetivo el pico real pasaría el límite,
    # loudnorm cambia solo a modo dinámico (comprime). Para evitarlo, salvo allow_dynamic,
    # se baja el objetivo hasta la ganancia máxima que respeta el pico (con un margen).
    target = float(ac["lufs"])
    in_i, in_tp = float(m1["input_i"]), float(m1["input_tp"])
    limit = float(ac["true_peak"]) - (float(ac.get("linear_margin", 0.2)) if margin is None else margin)
    if not ac.get("allow_dynamic", False) and in_tp + (target - in_i) > limit:
        target = round(in_i + (limit - in_tp), 2)
        base = f"loudnorm=I={target}:TP={ac['true_peak']}:LRA={ac['lra']}"
    second = (f"{base}:measured_I={m1['input_i']}:measured_TP={m1['input_tp']}"
              f":measured_LRA={m1['input_lra']}:measured_thresh={m1['input_thresh']}"
              f":offset={m1['target_offset']}:linear=true:print_format=json")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    r2 = run(["ffmpeg", "-y", "-hide_banner", "-nostats", *src_args, "-af", second,
              "-ar", str(ac["sample_rate"]), "-c:a", "pcm_f32le", str(out_wav)])
    m2 = json.loads(re.findall(r"\{[^{}]*\}", r2.stderr)[-1])
    info = {"pass1": m1, "pass2": m2, "type": m2.get("normalization_type"), "target_i": target}
    out_json.write_text(json.dumps(info, indent=1))
    return info


def normalize_song(cfg, s, force=False) -> dict:
    sp = song_paths(cfg, s)
    return loudnorm_two_pass(cfg, ["-i", str(sp["mp3"])], sp["norm_wav"], sp["norm_json"], force)


def measure_loudness(path: Path) -> tuple[float, float]:
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
             "-af", "ebur128=peak=true", "-f", "null", "-"])
    txt = r.stderr[r.stderr.rfind("Summary:"):]
    i = float(re.search(r"I:\s+(-?[\d.]+) LUFS", txt).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", txt).group(1))
    return i, tp


# ---------------------------------------------------------------------------
# Imagen: recorte vertical centrado en la cara
# ---------------------------------------------------------------------------
def face_box(cfg, image: Path):
    import cv2
    im = cv2.imread(str(image))
    H, W = im.shape[:2]
    model = P(cfg, "face_model")
    try:
        det = cv2.FaceDetectorYN.create(str(model), "", (W, H), 0.6, 0.3, 5000)
        _, faces = det.detect(im)
    except Exception as e:  # sin modelo o sin soporte: centro de la imagen
        print(f"  (detección de rostro no disponible: {e}; uso el centro)")
        faces = None
    if faces is None or len(faces) == 0:
        return None, W, H
    f = max(faces.tolist(), key=lambda f: f[2] * f[3])
    return (f[0], f[1], f[2], f[3]), W, H


def vertical_crop(cfg, image: Path, out_w: int, out_h: int):
    """Recorte out_w:out_h centrado en la cara (o en el centro si no hay cara).
    Devuelve (x0, y0, cw, ch, cara_encontrada, ancla_zoom, borde_sup_cara_en_px_de_salida)."""
    box, W, H = face_box(cfg, image)
    found = box is not None
    cx, cy = (box[0] + box[2] / 2, box[1] + box[3] / 2) if found else (W / 2, H / 2)
    ar = out_w / out_h
    if W / H > ar:
        ch, cw = H, int(round(H * ar / 2) * 2)
    else:
        cw, ch = W, int(round(W / ar / 2) * 2)
    x0 = int(min(max(cx - cw / 2, 0), W - cw))
    y0 = int(min(max(cy - ch / 2, 0), H - ch))
    anchor = ((cx - x0) / cw, (cy - y0) / ch)
    face_top = (box[1] - y0) * out_h / ch if found else None
    return x0, y0, cw, ch, found, anchor, face_top


# ---------------------------------------------------------------------------
# ASS
# ---------------------------------------------------------------------------
def ass_header(w: int, h: int, styles: list[str]) -> str:
    return "\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        *styles, "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]) + "\n"


def card_style(name, font, size, color):
    # sombra suave: borde transparente + sombra desenfocada (ver tags en card_event)
    return (f"Style: {name},{font},{size},{ass_color(color)},{ass_color(color)},&HFF000000,&H80000000,"
            f"0,0,0,0,100,100,0,0,1,2,3,2,0,0,0,1")


SOFT = r"\bord2\3a&HFF&\shad3\4c&H000000&\4a&H60&\blur5"


def card_event(start, end, style, x, y, text, fade_ms):
    return (f"Dialogue: 2,{fmt_ass_time(start)},{fmt_ass_time(end)},{style},,0,0,0,,"
            f"{{\\an2\\pos({x},{y})\\fad({fade_ms},{fade_ms}){SOFT}}}{text}")


def lyric_events(cfg, a, b, text, style="Lyrics") -> list[str]:
    """Línea de letra: halo oscuro muy difuso detrás (capa 0) + texto con borde (capa 1)."""
    lc = cfg["lyrics"]
    out = []
    if lc.get("halo_opacity", 0) > 0:
        ha = round((1 - lc["halo_opacity"]) * 255)
        out.append(f"Dialogue: 0,{fmt_ass_time(a)},{fmt_ass_time(b)},{style},,0,0,0,,"
                   f"{{\\1a&HFF&\\shad0\\bord{lc['halo_size']}\\3c&H000000&\\3a&H{ha:02X}&\\blur{lc['halo_size']}}}{text}")
    out.append(f"Dialogue: 1,{fmt_ass_time(a)},{fmt_ass_time(b)},{style},,0,0,0,,{{\\blur0.6}}{text}")
    return out


def lyric_style(cfg, name, size, margin_v, margin_lr):
    lc = cfg["lyrics"]
    return (f"Style: {name},{lc['font']},{size},{ass_color(lc['color'])},{ass_color(lc['color'])},"
            f"{ass_color(lc['outline_color'], lc['outline_opacity'])},&HA0000000,"
            f"0,0,0,0,100,100,0,0,1,{lc['outline']},1,2,{margin_lr},{margin_lr},{margin_v},1")


def write_single_ass(cfg, s, cues, path: Path, duration: float) -> tuple[float, float]:
    v, tc, lc = cfg["video"], cfg["title_card"], cfg["lyrics"]
    W, H = v["width"], v["height"]
    sy = H / 1080
    first = min((a for a, _, _, _ in cues), default=duration)
    card_end = min(tc["duration"], first - tc["min_gap_before_voice"])
    card_end = max(card_end, 1.2)
    fade = int(min(tc["fade"], card_end / 3) * 1000)
    styles = [
        card_style("Chapter", tc["chapter_font"], round(tc["chapter_size"] * sy), tc["color"]),
        card_style("Title", tc["title_font"], round(tc["title_size"] * sy), tc["color"]),
        card_style("Artist", tc["artist_font"], round(tc["artist_size"] * sy), tc["color"]),
        lyric_style(cfg, "Lyrics", round(lc["size"] * sy), round(lc["margin_bottom"] * sy),
                    round(lc["margin_sides"] * sy)),
    ]
    ev = [
        card_event(0, card_end, "Chapter", W // 2, round(tc["chapter_y"] * sy), f"Chapter {s['n']}", fade),
        card_event(0, card_end, "Title", W // 2, round(tc["title_y"] * sy), s["title"], fade),
        card_event(0, card_end, "Artist", W // 2, round(tc["artist_y"] * sy), cfg["artist"]["name"], fade),
    ]
    for a, b, t, _ in cues:
        ev += lyric_events(cfg, a, b, t)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ass_header(W, H, styles) + "\n".join(ev) + "\n", encoding="utf-8")
    return 0.0, card_end


def write_short_ass(cfg, s, cues, path: Path, duration: float, face_top: float | None = None):
    sc, tc = cfg["shorts"], cfg["title_card"]
    label_y = sc["label_y"]
    if face_top is not None:  # que el rótulo no tape la cara: lo sube si hace falta
        label_y = max(sc["label_min_y"], min(label_y, round(face_top - sc["label_size"] * 1.3 - sc["label_face_gap"])))
    W, H = sc["width"], sc["height"]
    styles = [
        card_style("Label", tc["chapter_font"], sc["label_size"], tc["color"]),
        lyric_style(cfg, "Lyrics", sc["lyrics_size"], sc["lyrics_margin_bottom"], sc["lyrics_margin_sides"]),
    ]
    label = f"{cfg['artist']['name']} · Chapter {s['n']}"
    ev = [f"Dialogue: 2,{fmt_ass_time(0)},{fmt_ass_time(duration + 1)},Label,,0,0,0,,"
          f"{{\\an8\\pos({W // 2},{label_y}){SOFT}}}{label}"]
    for a, b, t in cues:
        ev += lyric_events(cfg, a, b, t)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ass_header(W, H, styles) + "\n".join(ev) + "\n", encoding="utf-8")


def write_comp_card_ass(cfg, s, path: Path, seg_dur: float):
    v, tc, cc = cfg["video"], cfg["title_card"], cfg["compilation"]
    W, H = v["width"], v["height"]
    sy = H / 1080
    end = min(cc["card_duration"], seg_dur)
    fade = int(cc["card_fade"] * 1000)
    styles = [
        card_style("Chapter", tc["chapter_font"], round(tc["chapter_size"] * sy), tc["color"]),
        card_style("Title", tc["title_font"], round(tc["title_size"] * sy), tc["color"]),
    ]
    ev = [
        card_event(0, end, "Chapter", W // 2, round((tc["chapter_y"] + (tc["artist_y"] - tc["title_y"])) * sy),
                   f"Chapter {s['n']}", fade),
        card_event(0, end, "Title", W // 2, round(tc["artist_y"] * sy), s["title"], fade),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ass_header(W, H, styles) + "\n".join(ev) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Render de video
# ---------------------------------------------------------------------------
def esc_filter_path(p: Path) -> str:
    return rel(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def picture_chain(cfg, n_frames: int, out_w: int, out_h: int, crop: tuple | None = None,
                  anchor: tuple[float, float] | None = None) -> str:
    """Foto → B/N → 4K → zoom lento continuo → grano → viñeta.

    anchor = (ax, ay) en proporción del cuadro (0–1): el zoom se acerca hacia ese punto
    y lo mantiene fijo (se usa en los Shorts para que la cara no se mueva). Sin ancla: centro."""
    v = cfg["video"]
    k = v["zoom_source_scale"]
    sw, sh = out_w * k, out_h * k
    pre = ""
    if crop:
        x0, y0, cw, ch = crop
        pre = f"crop={cw}:{ch}:{x0}:{y0},"
    zs, ze = v["zoom_start"], v["zoom_end"]
    denom = max(1, n_frames - 1)
    if anchor:
        zx = f"{anchor[0]:.5f}*iw*(1-1/zoom)"
        zy = f"{anchor[1]:.5f}*ih*(1-1/zoom)"
    else:
        zx, zy = "iw/2-iw/zoom/2", "ih/2-ih/zoom/2"
    chain = (
        f"{pre}scale={sw}:{sh}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={sw}:{sh},setsar=1,format=yuv420p,hue=s=0,"
        f"zoompan=z='{zs}+({ze}-{zs})*on/{denom}':x='{zx}':y='{zy}'"
        f":d={n_frames}:s={out_w}x{out_h}:fps={v['fps']}"
    )
    if v["grain"] > 0:
        chain += f",noise=c0s={v['grain']}:c0f=t+u"
    chain += f",vignette=angle={v['vignette_angle']}"
    return chain


def x264_args(cfg) -> list[str]:
    v = cfg["video"]
    return ["-c:v", "libx264", "-preset", v["preset"], "-crf", str(v["crf"]), "-pix_fmt", "yuv420p",
            "-r", str(v["fps"]), "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv"]


def aac_args(cfg) -> list[str]:
    a = cfg["audio"]
    return ["-c:a", "aac", "-b:a", a["bitrate"], "-ar", str(a["sample_rate"])]


def render_single(cfg, s, cues, out: Path, limit: float | None = None):
    sp = song_paths(cfg, s)
    v = cfg["video"]
    dur = audio_samples_duration(sp["norm_wav"])
    n_frames = round(dur * v["fps"])
    write_single_ass(cfg, s, cues, sp["ass"], dur)
    vf = (picture_chain(cfg, n_frames, v["width"], v["height"])
          + f",subtitles=filename='{esc_filter_path(sp['ass'])}':fontsdir='{esc_filter_path(P(cfg, 'fonts_dir'))}'"
          + f",fade=t=out:st={dur - v['fade_out']:.3f}:d={v['fade_out']},format=yuv420p")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    t = limit if limit else dur
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", rel(sp["image"]), "-i", rel(sp["norm_wav"]),
           "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", "-map", "1:a",
           *x264_args(cfg), *aac_args(cfg), "-t", f"{t:.3f}", "-movflags", "+faststart", rel(tmp)]
    t0 = time.time()
    run(cmd)
    tmp.replace(out)
    return time.time() - t0


def short_window(timed: list[TimedLine], cfg, duration: float):
    """Ventana del Short: 1 s antes del primer estribillo hasta 1 s después de su última línea,
    ajustada a [min, max] segundos. Devuelve (ini, fin, nota)."""
    sc = cfg["shorts"]
    chorus = _first_chorus(timed)
    if not chorus:
        raise RuntimeError("no encontré el estribillo")
    a = max(0.0, timed[chorus[0]].start - sc["pad_before"])
    ends = [(k, timed[k].end + sc["pad_after"]) for k in range(chorus[-1], len(timed))]
    b = ends[0][1]
    note = ""
    if b - a < sc["min_duration"]:
        cand = next(((k, e) for k, e in ends if e - a >= sc["min_duration"]), None)
        if cand and cand[1] - a <= sc["max_duration"]:
            b = cand[1]
            extra = cand[0] - chorus[-1]
            note = f"el estribillo dura {ends[0][1] - a:.1f} s: se extiende {extra} línea(s) más para llegar a {sc['min_duration']:.0f} s"
        else:
            b = a + sc["min_duration"]
            note = f"el estribillo dura {ends[0][1] - a:.1f} s: se extiende hasta {sc['min_duration']:.0f} s"
    elif b - a > sc["max_duration"]:
        inside = [(k, timed[k].end + sc["pad_after"]) for k in chorus
                  if timed[k].end + sc["pad_after"] - a <= sc["max_duration"]]
        if inside and inside[-1][1] - a >= sc["min_duration"]:
            b = inside[-1][1]
            note = f"el estribillo dura {ends[0][1] - a:.1f} s: se corta en una línea antes de {sc['max_duration']:.0f} s"
        else:
            b = a + sc["max_duration"]
            note = f"el estribillo dura {ends[0][1] - a:.1f} s: se corta en {sc['max_duration']:.0f} s"
    b = min(b, duration)
    return a, b, note, chorus


def _first_chorus(timed: list[TimedLine]) -> list[int]:
    is_ch = [bool(re.search(r"chorus|coro|estribillo", t.section, re.I)) and not re.search(r"pre", t.section, re.I)
             for t in timed]
    if any(is_ch):
        k = is_ch.index(True)
        st = timed[k].stanza
        return [i for i, t in enumerate(timed) if t.stanza == st]
    # sin etiquetas: primera estrofa que se repite más adelante
    from collections import defaultdict
    stz = defaultdict(list)
    for i, t in enumerate(timed):
        stz[t.stanza].append(i)
    keys = sorted(stz)
    for ki, k in enumerate(keys):
        txt = [" ".join(norm_tokens(timed[i].text)) for i in stz[k]]
        for k2 in keys[ki + 1:]:
            txt2 = [" ".join(norm_tokens(timed[i].text)) for i in stz[k2]]
            if len(set(txt) & set(txt2)) >= max(1, len(txt) // 2):
                return stz[k]
    return stz[keys[0]] if keys else []


def timed_from_srt(cfg, s, lyrics_lines: list[LyricLine]) -> list[TimedLine]:
    """Con --use-srt: reconstruye tiempos por línea desde el .srt (para la ventana del Short)."""
    sp = song_paths(cfg, s)
    cues = read_srt(sp["srt"])
    lc = cfg["lyrics"]
    from rapidfuzz import fuzz
    out = []
    j = 0
    for k, ln in enumerate(lyrics_lines):
        best, bj = -1, None
        for jj in range(j, min(len(cues), j + 4)):
            sc = fuzz.ratio(" ".join(norm_tokens(ln.text)), " ".join(norm_tokens(cues[jj][2])))
            if sc > best:
                best, bj = sc, jj
        if bj is None:
            break
        a, b, _ = cues[bj]
        nxt = cues[bj + 1][0] if bj + 1 < len(cues) else 1e9
        end = b - lc["silence_hold"] if nxt - b > 0.05 else b + lc["lead"]
        out.append(TimedLine(k, ln.text, ln.section, ln.stanza, a + lc["lead"], max(end, a + 0.5), True,
                             show_start=a, show_end=b))
        j = bj + 1
    return out


def render_short(cfg, s, timed: list[TimedLine], cues_all, out: Path):
    sp = song_paths(cfg, s)
    sc, v = cfg["shorts"], cfg["video"]
    full = audio_samples_duration(sp["norm_wav"])
    a, b, note, chorus = short_window(timed, cfg, full)
    dur = b - a
    n_frames = round(dur * v["fps"])
    # solo las líneas que empiezan dentro de la ventana, con tiempos relativos
    cues = []
    for (cs, ce, t, _), tl in zip(cues_all, timed):
        if tl.start >= a - 0.01 and cs < b:
            cues.append((max(0, cs - a), min(dur, ce - a), t))
    x0, y0, cw, ch, found, anchor, face_top = vertical_crop(cfg, sp["image"], sc["width"], sc["height"])
    write_short_ass(cfg, s, cues, sp["short_ass"], dur, face_top)
    vf = (picture_chain(cfg, n_frames, sc["width"], sc["height"], crop=(x0, y0, cw, ch), anchor=anchor)
          + f",subtitles=filename='{esc_filter_path(sp['short_ass'])}':fontsdir='{esc_filter_path(P(cfg, 'fonts_dir'))}'"
          + ",format=yuv420p")
    af = (f"atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
          f"afade=t=in:st=0:d={sc['audio_fade_in']},afade=t=out:st={dur - sc['audio_fade_out']:.3f}:d={sc['audio_fade_out']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", rel(sp["image"]), "-i", rel(sp["norm_wav"]),
           "-filter_complex", f"[0:v]{vf}[v];[1:a]{af}[a]", "-map", "[v]", "-map", "[a]",
           *x264_args(cfg), *aac_args(cfg), "-t", f"{dur:.3f}", "-movflags", "+faststart", rel(tmp)]
    t0 = time.time()
    run(cmd)
    tmp.replace(out)
    return {"start": a, "end": b, "dur": dur, "note": note, "face": found, "crop": [x0, y0, cw, ch],
            "secs": time.time() - t0}


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------
def selected(cfg, only: str | None, kind: str):
    songs = [s for s in cfg["songs"] if s.get(kind)]
    if only:
        want = {int(x) for x in only.split(",")}
        songs = [s for s in songs if s["n"] in want]
    return songs


def cmd_align(cfg, args, quiet=False):
    lyrics = parse_lyrics_md(P(cfg, "lyrics_file"))
    report = {}
    for s in selected(cfg, args.only, "single"):
        sp = song_paths(cfg, s)
        if not quiet:
            print(f"[{sp['nn']}] {s['title']}: alineando …", flush=True)
        timed = align_song(cfg, s, lyrics)
        interp = [tl for tl in timed if not tl.located]
        report[s["n"]] = interp
        if not args.use_srt:
            write_srt(sp["srt"], [(a, b, ts) for a, b, _, ts in lyric_cues(cfg, s, timed, False)])
        if not quiet:
            print(f"      {len(timed) - len(interp)}/{len(timed)} líneas ubicadas"
                  + (f"; interpoladas: " + " | ".join(f"«{t.text}»" for t in interp) if interp else ""))
    return report


def cmd_preview(cfg, args):
    s = cfg["_songs"][args.song]
    sp = song_paths(cfg, s)
    lyrics = parse_lyrics_md(P(cfg, "lyrics_file"))
    timed = align_song(cfg, s, lyrics)
    if not args.use_srt:
        write_srt(sp["srt"], [(a, b, ts) for a, b, _, ts in lyric_cues(cfg, s, timed, False)])
    normalize_song(cfg, s)
    cues = lyric_cues(cfg, s, timed, args.use_srt)
    if args.use_srt:
        timed = timed_from_srt(cfg, s, lyrics[s["n"]])
    pdir = P(cfg, "output_dir") / "preview"
    out = pdir / f"preview-{sp['stem']}.mp4"
    secs = render_single(cfg, s, cues, out, limit=args.seconds)
    print(f"adelanto: {rel(out)} ({secs:.0f} s de render)")
    info = render_short(cfg, s, timed, cues, pdir / f"preview-short-{sp['nn']}.mp4")
    print(f"short:    {rel(pdir / f'preview-short-{sp['nn']}.mp4')} ({info['secs']:.0f} s de render) "
          f"ventana {info['start']:.2f}–{info['end']:.2f} ({info['dur']:.1f} s) {info['note']}")


def _single_job(cfg, s, use_srt):
    sp = song_paths(cfg, s)
    normalize_song(cfg, s)
    timed = load_alignment(cfg, s)
    cues = lyric_cues(cfg, s, timed, use_srt)
    secs = render_single(cfg, s, cues, sp["mp4"])
    print(f"[{sp['nn']}] sencillo listo: {rel(sp['mp4'])} ({secs / 60:.1f} min)", flush=True)


def cmd_singles(cfg, args):
    if not args.use_srt or not all(song_paths(cfg, s)["align"].exists() for s in selected(cfg, args.only, "single")):
        cmd_align(cfg, args)
    songs = selected(cfg, args.only, "single")
    for s in songs:
        normalize_song(cfg, s)
    with ThreadPoolExecutor(max_workers=cfg["video"]["jobs"]) as ex:
        list(ex.map(lambda s: _single_job(cfg, s, args.use_srt), songs))


def _short_job(cfg, s, use_srt, lyrics):
    sp = song_paths(cfg, s)
    normalize_song(cfg, s)
    timed = load_alignment(cfg, s)
    cues = lyric_cues(cfg, s, timed, use_srt)
    if use_srt:
        timed = timed_from_srt(cfg, s, lyrics[s["n"]])
    info = render_short(cfg, s, timed, cues, sp["short"])
    info_path = P(cfg, "work_dir") / "shorts" / f"short-{sp['nn']}.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=1))
    print(f"[{sp['nn']}] short listo: {rel(sp['short'])} ({info['dur']:.1f} s"
          + (f"; {info['note']}" if info["note"] else "") + ")", flush=True)


def cmd_shorts(cfg, args):
    songs = selected(cfg, args.only, "single")
    if not all(song_paths(cfg, s)["align"].exists() for s in songs):
        cmd_align(cfg, args)
    lyrics = parse_lyrics_md(P(cfg, "lyrics_file"))
    for s in songs:
        normalize_song(cfg, s)
    with ThreadPoolExecutor(max_workers=cfg["video"]["jobs"]) as ex:
        list(ex.map(lambda s: _short_job(cfg, s, args.use_srt, lyrics), songs))


def cmd_compilation(cfg, args):
    cc, v, ac = cfg["compilation"], cfg["video"], cfg["audio"]
    songs = [s for s in cfg["songs"] if s.get("compilation")]
    work = P(cfg, "work_dir") / "compilation"
    work.mkdir(parents=True, exist_ok=True)
    for s in songs:
        normalize_song(cfg, s)
    durs = [audio_samples_duration(song_paths(cfg, s)["norm_wav"]) for s in songs]
    # 1) audio: canciones (cada una ya a -14 LUFS) + 1,5 s de silencio, y el conjunto a -14 LUFS
    concat_wav = work / "vol.concat.wav"
    inputs, labels = [], []
    for k, s in enumerate(songs):
        inputs += ["-i", rel(song_paths(cfg, s)["norm_wav"])]
    fc = []
    idx = 0
    for k in range(len(songs)):
        labels.append(f"[{k}:a]")
        if k < len(songs) - 1:
            fc.append(f"anullsrc=r={ac['sample_rate']}:cl=stereo,atrim=duration={cc['gap']}[g{k}]")
            labels.append(f"[g{k}]")
    fc.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]")
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fc),
         "-map", "[out]", "-ar", str(ac["sample_rate"]), "-c:a", "pcm_f32le", rel(concat_wav)])
    norm_wav = work / "vol.norm.wav"
    # el margen solo hace falta en cada canción; el conjunto puede llegar justo a -1 dBTP
    info = loudnorm_two_pass(cfg, ["-i", rel(concat_wav)], norm_wav, work / "vol.loudnorm.json", force=True,
                             margin=0.0)
    total = audio_samples_duration(norm_wav)
    # 2) tramos de video
    starts, t = [], 0.0
    for k, d in enumerate(durs):
        starts.append(t)
        t += d + (cc["gap"] if k < len(durs) - 1 else 0)
    bounds = [round(x * v["fps"]) for x in starts] + [round(total * v["fps"])]
    seg_files = []

    def seg_job(k):
        s = songs[k]
        sp = song_paths(cfg, s)
        nf = bounds[k + 1] - bounds[k]
        seg_dur = nf / v["fps"]
        ass = work / f"card-{sp['nn']}.ass"
        write_comp_card_ass(cfg, s, ass, seg_dur)
        last = k == len(songs) - 1
        fo = v["fade_out"] if last else cc["segment_fade_out"]
        vf = (picture_chain(cfg, nf, v["width"], v["height"])
              + f",subtitles=filename='{esc_filter_path(ass)}':fontsdir='{esc_filter_path(P(cfg, 'fonts_dir'))}'"
              + f",fade=t=in:st=0:d={cc['segment_fade_in']},fade=t=out:st={seg_dur - fo:.3f}:d={fo},format=yuv420p")
        out = work / f"seg-{sp['nn']}.mp4"
        sig = work / f"seg-{sp['nn']}.sig"
        signature = json.dumps({"vf": vf, "x264": x264_args(cfg), "nf": nf,
                                "img": sp["image"].stat().st_mtime, "ass": ass.read_text()})
        if out.exists() and sig.exists() and sig.read_text() == signature:
            print(f"  tramo {sp['nn']} sin cambios: se reutiliza", flush=True)
            return out
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", rel(sp["image"]),
             "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", *x264_args(cfg),
             "-frames:v", str(nf), rel(out)])
        sig.write_text(signature)
        print(f"  tramo {sp['nn']} listo ({seg_dur:.1f} s)", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=v["jobs"]) as ex:
        seg_files = list(ex.map(seg_job, range(len(songs))))
    lst = work / "segments.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in seg_files))
    out = P(cfg, "output_dir") / f"{cc['output_name']}.mp4"
    tmp = out.with_suffix(".part.mp4")
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", rel(lst),
         "-i", rel(norm_wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy", *aac_args(cfg),
         "-t", f"{total:.3f}", "-movflags", "+faststart", rel(tmp)])
    tmp.replace(out)
    # 3) timestamps de capítulos
    lines = []
    for s, st in zip(songs, starts):
        sec = int(math.floor(st + 1e-6))
        stamp = f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if total >= 3600 else f"{sec // 60}:{sec % 60:02d}"
        lines.append(f"{stamp} {s['title']}")
    (P(cfg, "output_dir") / cc["timestamps_name"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"compilación lista: {rel(out)} ({total / 60:.1f} min, loudnorm {info['type']})")
    print("\n".join(lines))


def cmd_verify(cfg, args):
    rows = []
    out = P(cfg, "output_dir")
    items = []
    for s in cfg["songs"]:
        sp = song_paths(cfg, s)
        if s.get("single"):
            items.append((sp["mp4"], sp["norm_wav"], "sencillo"))
            info_p = P(cfg, "work_dir") / "shorts" / f"short-{sp['nn']}.json"
            if info_p.exists():
                items.append((sp["short"], json.loads(info_p.read_text())["dur"], "short"))
    cc = cfg["compilation"]
    items.append((out / f"{cc['output_name']}.mp4", P(cfg, "work_dir") / "compilation" / "vol.norm.wav", "compilación"))
    for mp4, ref, kind in items:
        if not mp4.exists():
            rows.append((rel(mp4), kind, None, None, None, None, None, "FALTA"))
            continue
        ref_d = ref if isinstance(ref, float) else audio_samples_duration(ref)
        r = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration:format=duration",
                 "-of", "json", str(mp4)])
        j = json.loads(r.stdout)
        vd = next(float(x["duration"]) for x in j["streams"] if x["codec_type"] == "video")
        ad = next(float(x["duration"]) for x in j["streams"] if x["codec_type"] == "audio")
        i, tp = measure_loudness(mp4)
        ok = abs(vd - ref_d) <= 0.1 and abs(ad - ref_d) <= 0.1
        rows.append((rel(mp4), kind, ref_d, vd, ad, i, tp, "OK" if ok else "REVISAR"))
    print(f"{'archivo':52s} {'tipo':11s} {'audio ref':>9s} {'video':>8s} {'audio':>8s} {'LUFS':>6s} {'dBTP':>6s}")
    for r in rows:
        if r[2] is None:
            print(f"{r[0]:52s} {r[1]:11s} {'—':>9s} {r[7]}")
            continue
        print(f"{r[0]:52s} {r[1]:11s} {r[2]:9.2f} {r[3]:8.2f} {r[4]:8.2f} {r[5]:6.1f} {r[6]:6.1f}  {r[7]}")
    (P(cfg, "work_dir") / "verify.json").write_text(json.dumps(rows, indent=1))
    return rows


def cmd_report(cfg, args):
    """Escribe output/REPORTE.md a partir de lo que dejaron los otros comandos en work/."""
    work, out = P(cfg, "work_dir"), P(cfg, "output_dir")
    vpath = work / "verify.json"
    rows = json.loads(vpath.read_text()) if vpath.exists() else cmd_verify(cfg, args)
    L = ["# Reporte de render", ""]
    L += ["## Archivos y verificación", "",
          "| Archivo | Tipo | Audio ref. (s) | Video (s) | Audio (s) | LUFS | dBTP | Estado |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        if r[2] is None:
            L.append(f"| {r[0]} | {r[1]} | — | — | — | — | — | {r[7]} |")
        else:
            L.append(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.1f} | {r[6]:.1f} | {r[7]} |")
    L += ["", "## Normalización (loudnorm, dos pasadas)", "",
          "| Canción | Entrada LUFS | Entrada dBTP | Objetivo LUFS | Modo |", "| --- | ---: | ---: | ---: | --- |"]
    for s in cfg["songs"]:
        jp = song_paths(cfg, s)["norm_json"]
        if jp.exists():
            j = json.loads(jp.read_text())
            L.append(f"| {s['n']:02d} {s['title']} | {j['pass1']['input_i']} | {j['pass1']['input_tp']} | "
                     f"{j.get('target_i', cfg['audio']['lufs'])} | {j['type']} |")
    cj = work / "compilation" / "vol.loudnorm.json"
    if cj.exists():
        j = json.loads(cj.read_text())
        L.append(f"| Compilación completa | {j['pass1']['input_i']} | {j['pass1']['input_tp']} | "
                 f"{j.get('target_i', cfg['audio']['lufs'])} | {j['type']} |")
    L += ["", "## Líneas interpoladas", ""]
    any_interp = False
    for s in cfg["songs"]:
        ap_ = song_paths(cfg, s)["align"]
        if s.get("single") and ap_.exists():
            bad = [d for d in json.loads(ap_.read_text()) if not d["located"]]
            for d in bad:
                any_interp = True
                L.append(f"- Cap. {s['n']}: «{d['text']}» ({fmt_srt_time(d['show_start'])})")
    if not any_interp:
        L.append("Ninguna: todas las líneas se ubicaron en la transcripción.")
    L += ["", "## Shorts", "", "| Short | Desde | Hasta | Duración | Cara | Nota |", "| --- | ---: | ---: | ---: | --- | --- |"]
    for s in cfg["songs"]:
        ip = work / "shorts" / f"short-{s['n']:02d}.json"
        if ip.exists():
            j = json.loads(ip.read_text())
            L.append(f"| short-{s['n']:02d} | {j['start']:.2f} | {j['end']:.2f} | {j['dur']:.1f} s | "
                     f"{'sí' if j['face'] else 'no (centro)'} | {j['note'] or '—'} |")
    tp = out / cfg["compilation"]["timestamps_name"]
    if tp.exists():
        L += ["", "## Capítulos de la compilación", "", "```text", tp.read_text().strip(), "```"]
    (out / "REPORTE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"reporte: {rel(out / 'REPORTE.md')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["align", "preview", "singles", "shorts", "compilation", "all", "verify", "report"])
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    ap.add_argument("--only", help="lista de capítulos, p. ej. 1,3,5")
    ap.add_argument("--use-srt", action="store_true", help="renderizar con los tiempos de output/NN-base.srt")
    ap.add_argument("--song", type=int, default=1, help="capítulo del adelanto")
    ap.add_argument("--seconds", type=float, default=43.0, help="duración del adelanto")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    if args.command == "align":
        cmd_align(cfg, args)
    elif args.command == "preview":
        cmd_preview(cfg, args)
    elif args.command == "singles":
        cmd_singles(cfg, args)
    elif args.command == "shorts":
        cmd_shorts(cfg, args)
    elif args.command == "compilation":
        cmd_compilation(cfg, args)
    elif args.command == "verify":
        cmd_verify(cfg, args)
    elif args.command == "report":
        cmd_report(cfg, args)
    elif args.command == "all":
        cmd_singles(cfg, args)
        cmd_shorts(cfg, args)
        cmd_compilation(cfg, args)
        cmd_verify(cfg, args)
        cmd_report(cfg, args)


if __name__ == "__main__":
    main()
