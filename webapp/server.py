#!/usr/bin/env python3
"""Tiny static server + upload endpoint for the handwriting seed pad.

- Serves index.html from this directory.
- POST /upload (raw ZIP) -> saves ZIP under HW_UPLOAD_DIR and extracts
  glyphs/<char>.png into HW_DATA_DIR (256x256 PNGs for zi2zi-JiT).
- GET /api/status -> counts for the UI.

Run:
    HW_DATA_DIR=/mnt/qnap_models/bambu-font-ai/data/user_glyphs \
    HW_UPLOAD_DIR=/mnt/qnap_models/bambu-font-ai/data/uploads \
    PORT=8088 python3 server.py
"""

import base64
import io
import json
import os
import re
import select
import subprocess
import sys
sys.path.insert(0, "/root/venvs/fontai/lib/python3.11/site-packages")
import threading
import time
from urllib.parse import parse_qs, urlparse, quote
import uuid
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "HW_DATA_DIR", "/mnt/qnap_models/bambu-font-ai/data/user_glyphs"))
UPLOAD_DIR = Path(os.environ.get(
    "HW_UPLOAD_DIR", "/mnt/qnap_models/bambu-font-ai/data/uploads"))
PORT = int(os.environ.get("PORT", "8088"))
AI_PY = os.environ.get("HW_AI_PY", "/root/venvs/fontai/bin/python")
AI_SCRIPT = WEB_DIR / "ai_generate.py"
AI_WORKER_SCRIPT = WEB_DIR / "ai_worker.py"
FONT_PY = os.environ.get("HW_FONT_PY", "/root/model-lab/venv/bin/python")
FONT_WORKER_SCRIPT = WEB_DIR / "font_worker.py"
AI_LOCK = threading.Lock()
AI_WORKERS = {}
SINGLE_ADAPTER = False
FONT_LOCK = threading.Lock()
FONT_WORKER = None
GLYPH_CACHE = None
STATE_FILE = Path(os.environ.get("HW_STATE_FILE", str(DATA_DIR.parent / "server_state.json")))
SEED_CHARS_FILE = Path(os.environ.get("HW_SEED_CHARS", str(DATA_DIR.parent / "seed_chars_2026.txt")))
STATE_LOCK = threading.Lock()
SERVER_STATE = None

# ---- font 1: brush image intake ----
FONT1_DIR = Path(os.environ.get("HW_FONT1_DIR", str(DATA_DIR.parent / "font1")))
FONT1_CROPS = FONT1_DIR / "crops"
FONT1_SRC = FONT1_DIR / "src"
FONT1_STATE = FONT1_DIR / "variants.json"
FONT1_PY = os.environ.get("HW_FONT1_PY", "/root/venvs/fontai/bin/python")
FONT1_INGEST = WEB_DIR / "font1_ingest.py"
FONTGEN_DIR = DATA_DIR.parent / "fontgen"
FONTGEN_STATE = FONTGEN_DIR / "state.json"
FONTGEN_BUILD = WEB_DIR / "fontgen_build.py"
FONTGEN_JOBS_DIR = FONTGEN_DIR / "jobs"
FONTGEN_JOBS = {}
FONTGEN_JOB_LOCK = threading.Lock()
FONT1_LOCK = threading.Lock()

GLYPH_NAME = re.compile(r"^glyphs/([^\x00-\x1f/\\]{1,4})\.png$")


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, fmt, *args):
        # keep the console quiet; uploads are logged explicitly
        pass

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _glyph_payload(self):
        global GLYPH_CACHE
        if GLYPH_CACHE is not None:
            return GLYPH_CACHE
        chars = {}
        canvas = 256
        font = "myhand"
        zips = sorted(UPLOAD_DIR.glob("upload-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        for zp in zips:
            try:
                with zipfile.ZipFile(zp) as zf:
                    if "strokes.json" not in zf.namelist():
                        continue
                    data = json.loads(zf.read("strokes.json").decode("utf-8"))
                    canvas = data.get("canvas", canvas)
                    font = data.get("font", font)
                    for item in data.get("characters", []):
                        ch = item.get("char")
                        strokes = item.get("strokes") or []
                        if isinstance(ch, str) and len(ch) == 1 and strokes and ch not in chars:
                            chars[ch] = {"char": ch, "savedAt": item.get("savedAt"), "strokes": strokes}
            except Exception as e:
                print(f"[restore] skip {zp.name}: {e}")
        GLYPH_CACHE = {
            "ok": True, "count": len(chars), "canvas": canvas, "font": font,
            "characters": list(chars.values()),
        }
        return GLYPH_CACHE

    def _load_state(self):
        global SERVER_STATE
        if SERVER_STATE is not None:
            return SERVER_STATE
        if STATE_FILE.exists():
            try:
                SERVER_STATE = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                return SERVER_STATE
            except Exception as e:
                print(f"[state] load failed: {e}")
        chars = []
        if SEED_CHARS_FILE.exists():
            chars = [c for c in SEED_CHARS_FILE.read_text(encoding="utf-8").strip() if not c.isspace()]
        base = self._glyph_payload()
        strokes = {}
        saved = {}
        for item in base.get("characters", []):
            ch = item.get("char")
            if ch:
                strokes[ch] = item.get("strokes", [])
                saved[ch] = item.get("savedAt") or 0
                if ch not in chars:
                    chars.append(ch)
        SERVER_STATE = {"chars": chars, "strokes": strokes, "saved": saved,
                        "canvas": base.get("canvas", 256), "font": base.get("font", "myhand")}
        return SERVER_STATE

    def _persist_state(self):
        if SERVER_STATE is None:
            return
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(SERVER_STATE, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATE_FILE)

    def _state_payload(self):
        st = self._load_state()
        return {"ok": True, "chars": st["chars"], "strokes": st["strokes"],
                "saved": st["saved"], "canvas": st.get("canvas", 256), "font": st.get("font", "myhand")}

    def _save_glyph(self):
        global GLYPH_CACHE
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"ok": False, "error": "bad content-length"})
        if length <= 0 or length > 8 * 1024 * 1024:
            return self._json(400, {"ok": False, "error": "empty or too large"})
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            ch = str(data.get("char", ""))
            strokes = data.get("strokes") or []
            png = data.get("png", "")
            if len(ch) != 1 or not strokes:
                raise ValueError("bad char/strokes")
            if png.startswith("data:image/png;base64,"):
                png = png.split(",", 1)[1]
            raw = base64.b64decode(png, validate=True)
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("not png")
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad payload: {e}"})
        with STATE_LOCK:
            st = self._load_state()
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / f"{ch}.png").write_bytes(raw)
            if ch not in st["chars"]:
                st["chars"].append(ch)
            st["strokes"][ch] = strokes
            st["saved"][ch] = data.get("savedAt") or int(time.time() * 1000)
            self._persist_state()
            GLYPH_CACHE = None
        return self._json(200, {"ok": True, "char": ch, "count": len(st["chars"])})

    # ---------------- font 1: brush scans ----------------
    def _font1_load(self):
        if FONT1_STATE.exists():
            try:
                return json.loads(FONT1_STATE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"variants": []}

    def _font1_persist(self, st):
        FONT1_DIR.mkdir(parents=True, exist_ok=True)
        tmp = FONT1_STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(FONT1_STATE)

    def _font1_payload(self, st=None):
        st = st or self._font1_load()
        variants = []
        for v in st.get("variants", []):
            variants.append({**v, "url": f"/font1-img/{v['id']}"})
        recognized = sum(1 for v in variants if v.get("char") not in (None, "", "?"))
        selected = sum(1 for v in variants if v.get("selected") and v.get("char") not in (None, "", "?"))
        return {"ok": True, "variants": variants, "count": len(variants),
                "recognized": recognized, "selectedRecognized": selected}

    def _font1_body(self, limit=120 * 1024 * 1024):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > limit:
            raise ValueError("empty or too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _font1_upload(self):
        try:
            payload = self._font1_body()
            files = payload.get("files") or []
            if not isinstance(files, list) or not files:
                raise ValueError("no files")
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad upload: {e}"})
        FONT1_CROPS.mkdir(parents=True, exist_ok=True)
        FONT1_SRC.mkdir(parents=True, exist_ok=True)
        st = self._font1_load()
        added = []
        errors = []
        for item in files[:30]:
            try:
                name = str(item.get("name") or "upload")
                data = str(item.get("data") or "")
                if "," in data and data.startswith("data:"):
                    data = data.split(",", 1)[1]
                raw = base64.b64decode(data, validate=True)
                ext = ".jpg" if raw[:2] == b"\xff\xd8" else ".png"
                sid = uuid.uuid4().hex[:12]
                src = FONT1_SRC / f"{sid}{ext}"
                src.write_bytes(raw)
                jout = FONT1_CROPS / f"{sid}.json"
                proc = subprocess.run(
                    [FONT1_PY, str(FONT1_INGEST), "--image", str(src),
                     "--out-dir", str(FONT1_CROPS), "--json-out", str(jout)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
                if proc.returncode != 0:
                    raise RuntimeError((proc.stderr or proc.stdout)[-300:])
                items = json.loads(jout.read_text(encoding="utf-8"))
                for it in items:
                    vid = it["id"]
                    ch = it.get("char") or "?"
                    st["variants"].append({
                        "id": vid, "char": ch, "selected": False,
                        "crop": f"crops/{vid}.png", "src": f"src/{src.name}",
                        "created": int(time.time() * 1000), "sourceName": name,
                        "size": it.get("size"),
                    })
                    added.append({"id": vid, "char": ch})
                try:
                    jout.unlink()
                except Exception:
                    pass
            except Exception as e:
                errors.append(f"{item.get('name', '?')}: {e}")
        self._font1_persist(st)
        return self._json(200, {"ok": True, "added": added, "count": len(added),
                                "errors": errors, **{k: v for k, v in self._font1_payload(st).items() if k != "ok"}})

    def _font1_update(self, action):
        try:
            payload = self._font1_body(2 * 1024 * 1024)
            vid = str(payload.get("id") or "")
            if not vid:
                raise ValueError("no id")
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad payload: {e}"})
        with FONT1_LOCK:
            st = self._font1_load()
            target = next((v for v in st["variants"] if v.get("id") == vid), None)
            if target is None:
                return self._json(404, {"ok": False, "error": "variant not found"})
            if action == "label":
                ch = str(payload.get("char") or "?").strip()
                if ch != "?" and len(ch) != 1:
                    return self._json(400, {"ok": False, "error": "char must be one character or ?"})
                target["char"] = ch or "?"
                if ch == "?":
                    target["selected"] = False
            elif action == "select":
                ch = target.get("char")
                for v in st["variants"]:
                    if v.get("char") == ch and ch not in (None, "", "?"):
                        v["selected"] = (v.get("id") == vid)
                target["selected"] = True
            elif action == "delete":
                st["variants"] = [v for v in st["variants"] if v.get("id") != vid]
            else:
                return self._json(400, {"ok": False, "error": "unknown action"})
            self._font1_persist(st)
            return self._json(200, self._font1_payload(st))

    def _font1_serve(self, vid):
        st = self._font1_load()
        target = next((v for v in st["variants"] if v.get("id") == vid), None)
        if target is None:
            return self.send_error(404)
        path = FONT1_DIR / target.get("crop", "")
        try:
            raw = path.read_bytes()
        except Exception:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/status":
            count = len(list(DATA_DIR.glob("*.png"))) if DATA_DIR.exists() else 0
            return self._json(200, {"ok": True, "glyphs": count, "dir": str(DATA_DIR)})
        if path == "/api/glyphs":
            return self._json(200, self._glyph_payload())
        if path == "/api/state":
            return self._json(200, self._state_payload())
        if path == "/api/font1/state":
            return self._json(200, self._font1_payload())
        if path == "/api/fontgen/state":
            qs = parse_qs(urlparse(self.path).query)
            style = (qs.get("style") or ["hand"])[0]
            source = (qs.get("source") or ["mine"])[0]
            return self._json(200, {"ok": True, "style": style, "source": source, "chars": self._fontgen_chars(style, source)})
        if path == "/api/fontgen/preview":
            return self._fontgen_preview()
        if path == "/api/fontgen/download":
            return self._fontgen_download()
        if path == "/api/fontgen/batch/status":
            return self._fontgen_batch_status()
        if path.startswith("/font1-img/"):
            return self._font1_serve(path.split("/font1-img/", 1)[1])
        return super().do_GET()

    def _ensure_worker(self, font=0):
        global AI_WORKERS
        key = "adapter" if SINGLE_ADAPTER else font
        w = AI_WORKERS.get(key)
        if w is not None and w.poll() is None:
            return w
        env = dict(os.environ)
        env["HW_AI_ORIGINAL_BASE"] = "0"
        env["HW_AI_BEST_N"] = "1"
        if SINGLE_ADAPTER:
            env["HW_AI_ADAPTER"] = "1"
        else:
            env["HW_AI_FONT"] = str(font)
        log = open(f"/tmp/ai_worker_font{font}.log", "ab")
        w = subprocess.Popen(
            [AI_PY, "-u", str(AI_WORKER_SCRIPT)],
            cwd=str(WEB_DIR), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log, text=True, bufsize=1, env=env,
        )
        AI_WORKERS[key] = w
        return w

    def _ensure_font_worker(self):
        global FONT_WORKER
        if FONT_WORKER is not None and FONT_WORKER.poll() is None:
            return FONT_WORKER
        log = open("/tmp/font_worker.log", "ab")
        FONT_WORKER = subprocess.Popen(
            [FONT_PY, "-u", str(FONT_WORKER_SCRIPT)],
            cwd=str(WEB_DIR), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log, text=True, bufsize=1,
        )
        return FONT_WORKER

    def _font_call(self, payload, timeout=900):
        with FONT_LOCK:
            worker = self._ensure_font_worker()
            worker.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            worker.stdin.flush()
            while True:
                ready, _, _ = select.select([worker.stdout], [], [], timeout)
                if not ready:
                    return {"ok": False, "error": "fontdiffuser timeout"}
                line = worker.stdout.readline()
                if not line:
                    return {"ok": False, "error": "fontdiffuser closed stdout"}
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except Exception:
                    print("[font] non-json:", line[:160])
                    continue

    def _ai_glyph(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"ok": False, "error": "bad content-length"})
        if length <= 0 or length > 16 * 1024:
            return self._json(400, {"ok": False, "error": "empty or too large"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            chars = [str(c) for c in payload.get("chars", []) if str(c).strip()]
            model = payload.get("model", "zi2zi")
            font = 1 if int(payload.get("font", 0) or 0) == 1 else 0
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad json: {e}"})
        chars = list(dict.fromkeys(chars))[:20]
        if not chars:
            return self._json(400, {"ok": False, "error": "no chars"})
        if model == "fontdiffuser":
            data = self._font_call({"chars": chars})
            if not data.get("ok"):
                return self._json(500, {"ok": False, "error": data.get("error", "fontdiffuser failed")})
            return self._json(200, {"ok": True, "strokes": data.get("strokes", {})})
        try:
            with AI_LOCK:
                worker = self._ensure_worker(font)
                worker.stdin.write(json.dumps({"chars": chars, "font": font}, ensure_ascii=False) + "\n")
                worker.stdin.flush()
                ready, _, _ = select.select([worker.stdout], [], [], 600)
                if not ready:
                    return self._json(500, {"ok": False, "error": "worker timeout"})
                line = worker.stdout.readline()
            data = json.loads(line)
            if not data.get("ok"):
                return self._json(500, {"ok": False, "error": data.get("error", "worker failed")})
            return self._json(200, {"ok": True, "strokes": data.get("strokes", {})})
        except Exception as e:
            global AI_WORKERS
            key = "adapter" if SINGLE_ADAPTER else font
            w = AI_WORKERS.pop(key, None)
            try:
                if w and w.poll() is None:
                    w.kill()
            except Exception:
                pass
            return self._json(500, {"ok": False, "error": str(e)})

    def _ai_variants(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"ok": False, "error": "bad content-length"})
        if length <= 0 or length > 16 * 1024:
            return self._json(400, {"ok": False, "error": "empty or too large"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            chars = [str(c) for c in payload.get("chars", []) if str(c).strip()]
            model = payload.get("model", "zi2zi")
            font = 1 if int(payload.get("font", 0) or 0) == 1 else 0
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad json: {e}"})
        chars = chars[:20]
        if not chars:
            return self._json(400, {"ok": False, "error": "no chars"})
        if model == "fontdiffuser":
            data = self._font_call({"chars": chars, "variants": True})
            if not data.get("ok"):
                return self._json(500, {"ok": False, "error": data.get("error", "fontdiffuser failed")})
            return self._json(200, {"ok": True, "variants": data.get("variants", {}), "images": data.get("images", {})})
        try:
            with AI_LOCK:
                worker = self._ensure_worker(font)
                worker.stdin.write(json.dumps({"chars": chars, "variants": True, "font": font}, ensure_ascii=False) + "\n")
                worker.stdin.flush()
                ready, _, _ = select.select([worker.stdout], [], [], 600)
                if not ready:
                    return self._json(500, {"ok": False, "error": "worker timeout"})
                line = worker.stdout.readline()
            data = json.loads(line)
            if not data.get("ok"):
                return self._json(500, {"ok": False, "error": data.get("error", "worker failed")})
            return self._json(200, {"ok": True, "variants": data.get("variants", {}), "images": data.get("images", {})})
        except Exception as e:
            global AI_WORKERS
            key = "adapter" if SINGLE_ADAPTER else font
            w = AI_WORKERS.pop(key, None)
            try:
                if w and w.poll() is None:
                    w.kill()
            except Exception:
                pass
            return self._json(500, {"ok": False, "error": str(e)})

    def _fontgen_load(self):
        try:
            return json.loads(FONTGEN_STATE.read_text(encoding="utf-8"))
        except Exception:
            return {"hand": {}, "brush": {}}

    def _fontgen_save(self, st):
        FONTGEN_DIR.mkdir(parents=True, exist_ok=True)
        tmp = FONTGEN_STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        tmp.replace(FONTGEN_STATE)

    @staticmethod
    def _fontgen_std_chars(source):
        chars = []
        seen = set()
        def add(ch):
            if ch not in seen:
                seen.add(ch); chars.append(ch)
        if source == "both":
            base = Handler._fontgen_std_chars("gb2312")
            from opencc import OpenCC
            cc = OpenCC("s2t")
            for ch in base:
                add(ch)
                t = cc.convert(ch)
                if t != ch:
                    add(t)
            return chars
        if source == "gb2312":
            for b1 in range(0xB0, 0xF8):
                for b2 in range(0xA1, 0xFF):
                    try:
                        ch = bytes([b1, b2]).decode("gb2312")
                    except Exception:
                        continue
                    if "\u4e00" <= ch <= "\u9fff":
                        add(ch)
        elif source == "big5":
            for b1 in range(0xA4, 0xFA):
                for b2 in range(0x40, 0xFF):
                    if b2 in (0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F, 0xA0):
                        continue
                    try:
                        ch = bytes([b1, b2]).decode("big5")
                    except Exception:
                        continue
                    if "\u4e00" <= ch <= "\u9fff":
                        add(ch)
        return chars

    def _fontgen_chars(self, style, source="mine"):
        fst = self._fontgen_load()
        over = fst.get(style, {})
        if style == "hand":
            main = self._load_state()
            chars = list(main.get("chars", []))
            if source in ("gb2312", "big5", "both"):
                std = self._fontgen_std_chars(source)
                chars = list(dict.fromkeys(std + chars))
            main_strokes = main.get("strokes", {})
            return [{"char": c, "approved": bool(over.get(c, {}).get("approved", True)),
                     "missing": not bool(over.get(c, {}).get("strokes") or main_strokes.get(c)),
                     "preview": f"/api/fontgen/preview?style=hand&ch={quote(c)}"} for c in chars]
        f1 = self._font1_load()
        rows = []
        for v in f1.get("variants", []):
            ch = v.get("char")
            if not v.get("selected") or ch in (None, "", "?"):
                continue
            rows.append({"char": ch, "approved": bool(over.get(ch, {}).get("approved", True)),
                         "missing": not (FONTGEN_DIR / "brush" / f"{ch}.png").exists(),
                         "preview": f"/api/fontgen/preview?style=brush&ch={quote(ch)}"})
        if source in ("gb2312", "big5", "both"):
            have = {r["char"] for r in rows}
            for ch in self._fontgen_std_chars(source):
                if ch not in have:
                    rows.append({"char": ch, "approved": False, "missing": True, "preview": f"/api/fontgen/preview?style=brush&ch={quote(ch)}"})
        return rows

    @staticmethod
    def _fontgen_punct():
        return set("，。！？；：、（）《》“”‘’—…·「」『』【】〔〕,.!?;:()[]{}<>\"'")

    def _fontgen_preview(self):
        qs = parse_qs(urlparse(self.path).query)
        style = (qs.get("style") or ["hand"])[0]
        ch = (qs.get("ch") or [""])[0]
        if not ch:
            return self._json(400, {"ok": False, "error": "no char"})
        if style == "brush":
            p = FONTGEN_DIR / "brush" / f"{ch}.png"
            if not p.exists():
                f1 = self._font1_load()
                for v in f1.get("variants", []):
                    if v.get("selected") and v.get("char") == ch:
                        p = FONT1_DIR / v["crop"]
                        break
            if not p.exists():
                body = f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="#fff"/><text x="128" y="150" font-size="110" text-anchor="middle" fill="#ddd">{ch}</text></svg>'.encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "image/svg+xml"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            body = p.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "image/png"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        st = self._load_state()
        fst = self._fontgen_load()
        strokes = (fst.get("hand", {}).get(ch, {}) or {}).get("strokes") or st.get("strokes", {}).get(ch) or []
        if not strokes:
            body = f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="#fff"/><text x="128" y="150" font-size="110" text-anchor="middle" fill="#ddd">{ch}</text></svg>'.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "image/svg+xml"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        parts = []
        for s in strokes:
            pts = []
            for p in s:
                if isinstance(p, (list, tuple)): x, y = p[0], p[1]
                else: x, y = p.get("x", 0), p.get("y", 0)
                pts.append(f"{float(x):.2f},{float(y):.2f}")
            if len(pts) == 1: pts.append(pts[0])
            if len(pts) >= 2:
                parts.append(f'<path d="M {" L ".join(pts)}" fill="none" stroke="#000" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>')
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" fill="#fff"/>{"".join(parts)}</svg>'
        body = svg.encode("utf-8")
        self.send_response(200); self.send_header("Content-Type", "image/svg+xml; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _fontgen_regen_one(self, style, ch):
        if len(ch) != 1: raise ValueError("bad char")
        font = 1 if style == "brush" else 0
        with AI_LOCK:
            worker = self._ensure_worker(font)
            worker.stdin.write(json.dumps({"chars": [ch], "variants": True, "font": font}, ensure_ascii=False) + "\n")
            worker.stdin.flush()
            ready, _, _ = select.select([worker.stdout], [], [], 600)
            if not ready: raise RuntimeError("worker timeout")
            line = worker.stdout.readline()
        data = json.loads(line)
        if not data.get("ok"): raise RuntimeError(data.get("error", "worker failed"))
        st = self._fontgen_load()
        over = st.setdefault(style, {})
        if style == "brush":
            img = (data.get("images") or {}).get(ch)
            if not img: raise RuntimeError("no image")
            b64 = img.split(",", 1)[1]
            (FONTGEN_DIR / "brush").mkdir(parents=True, exist_ok=True)
            (FONTGEN_DIR / "brush" / f"{ch}.png").write_bytes(base64.b64decode(b64))
            over[ch] = {"approved": True}
        else:
            vs = (data.get("variants") or {}).get(ch) or []
            if not vs: raise RuntimeError("no strokes")
            over[ch] = {"approved": True, "strokes": vs[0]}
        st[style] = over; self._fontgen_save(st)
        return f"/api/fontgen/preview?style={style}&ch={quote(ch)}"

    def _fontgen_regen(self):
        try:
            payload = self._font1_body(2 * 1024 * 1024)
            style = payload.get("style", "hand")
            ch = str(payload.get("char", ""))
        except Exception as e:
            return self._json(400, {"ok": False, "error": str(e)})
        try:
            preview = self._fontgen_regen_one(style, ch)
            return self._json(200, {"ok": True, "preview": preview})
        except Exception as e:
            return self._json(500, {"ok": False, "error": str(e)})

    def _fontgen_job_path(self, jid):
        return FONTGEN_JOBS_DIR / f"{jid}.json"

    def _fontgen_job_save(self, job):
        FONTGEN_JOBS_DIR.mkdir(parents=True, exist_ok=True)
        p = self._fontgen_job_path(job["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def _fontgen_job_load(self, jid):
        p = self._fontgen_job_path(jid)
        if not p.exists(): return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _fontgen_batch_start(self):
        try:
            payload = self._font1_body(4 * 1024 * 1024)
            style = payload.get("style", "hand")
            source = payload.get("source", "mine")
        except Exception as e:
            return self._json(400, {"ok": False, "error": str(e)})
        with FONTGEN_JOB_LOCK:
            for j in FONTGEN_JOBS.values():
                if j.get("status") in ("running", "stopping"):
                    return self._json(409, {"ok": False, "error": "已有批量任务在运行", "job": j.get("id")})
            chars = [x["char"] for x in self._fontgen_chars(style, source)]
            job = {"id": time.strftime("%Y%m%d-%H%M%S"), "style": style, "source": source,
                   "chars": chars, "index": 0, "done": 0, "failed": 0,
                   "status": "running", "current": "", "started": time.time(), "updated": time.time(), "items": []}
            FONTGEN_JOBS[job["id"]] = job
            self._fontgen_job_save(job)
        threading.Thread(target=self._fontgen_batch_run, args=(job["id"],), daemon=True).start()
        return self._json(200, {"ok": True, "job": job["id"], "total": len(chars)})

    def _fontgen_batch_run(self, jid):
        job = self._fontgen_job_load(jid)
        if not job: return
        chars = job.get("chars", [])
        while job.get("index", 0) < len(chars):
            disk = self._fontgen_job_load(jid)
            if disk and disk.get("status") == "stopping":
                job["status"] = "cancelled"; job["updated"] = time.time(); self._fontgen_job_save(job); return
            i = job["index"]; ch = chars[i]
            job["current"] = ch; job["updated"] = time.time(); self._fontgen_job_save(job)
            try:
                preview = self._fontgen_regen_one(job["style"], ch)
                job["done"] = job.get("done", 0) + 1
                job["items"] = (job.get("items", []) + [{"char": ch, "preview": preview, "updated": time.time()}])[-200:]
            except Exception:
                job["failed"] = job.get("failed", 0) + 1
            job["index"] = i + 1; job["updated"] = time.time()
            self._fontgen_job_save(job)
        job["status"] = "done"; job["current"] = ""; job["updated"] = time.time(); self._fontgen_job_save(job)

    def _fontgen_batch_status(self):
        qs = parse_qs(urlparse(self.path).query); jid = (qs.get("job") or [""])[0]
        job = self._fontgen_job_load(jid)
        if not job: return self._json(404, {"ok": False, "error": "job not found"})
        out = {k: v for k, v in job.items() if k != "chars"}
        out["total"] = len(job.get("chars", []))
        out["ok"] = True
        return self._json(200, out)

    def _fontgen_batch_stop(self):
        try:
            payload = self._font1_body(1024 * 1024); jid = str(payload.get("job", ""))
        except Exception as e:
            return self._json(400, {"ok": False, "error": str(e)})
        job = self._fontgen_job_load(jid)
        if not job: return self._json(404, {"ok": False, "error": "job not found"})
        if job.get("status") == "running":
            job["status"] = "stopping"; job["updated"] = time.time(); self._fontgen_job_save(job)
        return self._json(200, {"ok": True, "job": jid, "status": job.get("status")})

    def _fontgen_resume_jobs(self):
        FONTGEN_JOBS_DIR.mkdir(parents=True, exist_ok=True)
        for p in FONTGEN_JOBS_DIR.glob("*.json"):
            try:
                job = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            FONTGEN_JOBS[job["id"]] = job
            if job.get("status") == "stopping":
                job["status"] = "cancelled"; job["updated"] = time.time()
                try:
                    self._fontgen_job_save(job)
                except Exception:
                    pass
            elif job.get("status") == "running":
                threading.Thread(target=self._fontgen_batch_run, args=(job["id"],), daemon=True).start()

    def _fontgen_approve(self):
        try:
            payload = self._font1_body(1024 * 1024)
            style = payload.get("style", "hand"); ch = str(payload.get("char", ""))
            if len(ch) != 1: raise ValueError("bad char")
            st = self._fontgen_load(); over = st.setdefault(style, {})
            over.setdefault(ch, {})["approved"] = bool(payload.get("approved", True))
            st[style] = over; self._fontgen_save(st)
            return self._json(200, {"ok": True})
        except Exception as e:
            return self._json(400, {"ok": False, "error": str(e)})

    def _fontgen_build(self):
        try:
            payload = self._font1_body(2 * 1024 * 1024)
            style = payload.get("style", "hand")
            source = payload.get("source", "mine")
            # output follows selected charset automatically
            charset = "t" if source == "big5" else ("both" if source == "both" else "s")
            fmt = "single" if payload.get("format") == "single" and style == "hand" else "ttf"
        except Exception as e:
            return self._json(400, {"ok": False, "error": str(e)})
        try:
            from opencc import OpenCC
            cc = OpenCC("s2t") if charset in ("t", "both") else None
            st = self._load_state(); fst = self._fontgen_load(); over = fst.get(style, {})
            glyphs = []
            source_chars = list(st.get("chars", []))
            if source in ("gb2312", "big5", "both"):
                source_chars = list(dict.fromkeys(self._fontgen_std_chars(source) + source_chars))
            if style == "hand":
                for ch in source_chars:
                    if over.get(ch, {}).get("approved") is False: continue
                    src = over.get(ch, {}).get("strokes") or st.get("strokes", {}).get(ch)
                    if not src: continue
                    target = cc.convert(ch) if cc else ch
                    if charset == "both":
                        glyphs.append({"char": ch, "strokes": src})
                        tsrc = over.get(target, {}).get("strokes") or src
                        if target != ch:
                            glyphs.append({"char": target, "strokes": tsrc})
                    else:
                        glyphs.append({"char": target, "strokes": src})
            else:
                f1 = self._font1_load()
                have = set()
                for v in f1.get("variants", []):
                    ch = v.get("char")
                    if not v.get("selected") or ch in (None, "", "?"): continue
                    have.add(ch)
                    if over.get(ch, {}).get("approved") is False: continue
                    img = FONTGEN_DIR / "brush" / f"{ch}.png"
                    if not img.exists(): img = FONT1_DIR / v["crop"]
                    if not img.exists(): continue
                    target = cc.convert(ch) if cc else ch
                    if charset == "both":
                        glyphs.append({"char": ch, "image": str(img)})
                        timg = FONTGEN_DIR / "brush" / f"{target}.png"
                        if target != ch and timg.exists():
                            glyphs.append({"char": target, "image": str(timg)})
                        elif target != ch:
                            glyphs.append({"char": target, "image": str(img)})
                    else:
                        glyphs.append({"char": target, "image": str(img)})
                if source in ("gb2312", "big5", "both"):
                    for ch in self._fontgen_std_chars(source):
                        if ch in have or ch in self._fontgen_punct(): continue
                        img = FONTGEN_DIR / "brush" / f"{ch}.png"
                        if not img.exists() or over.get(ch, {}).get("approved") is False: continue
                        target = cc.convert(ch) if cc else ch
                        glyphs.append({"char": target, "image": str(img)})
            # dedupe target chars
            seen = set(); uniq = []
            for g in glyphs:
                if g["char"] in seen: continue
                seen.add(g["char"]); uniq.append(g)
            if not uniq:
                return self._json(400, {"ok": False, "error": "no approved glyphs"})
            FONTGEN_DIR.mkdir(parents=True, exist_ok=True)
            ext = "zip" if fmt == "single" else "ttf"
            name = f"myhand_{style}_{charset}.{ext}"
            out = FONTGEN_DIR / name
            inp = {"style": style, "charset": charset, "format": fmt, "out": str(out), "glyphs": uniq}
            ip = FONTGEN_DIR / "build_input.json"
            ip.write_text(json.dumps(inp, ensure_ascii=False), encoding="utf-8")
            proc = subprocess.run([str(Path("/root/venvs/fontai/bin/python")), str(FONTGEN_BUILD), str(ip)], capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                return self._json(500, {"ok": False, "error": proc.stderr[-800:] or "font build failed"})
            return self._json(200, {"ok": True, "file": name, "glyphs": len(uniq), "download": f"/api/fontgen/download?name={quote(name)}"})
        except Exception as e:
            return self._json(500, {"ok": False, "error": str(e)})

    def _fontgen_download(self):
        qs = parse_qs(urlparse(self.path).query); name = (qs.get("name") or [""])[0]
        p = (FONTGEN_DIR / name).resolve()
        if not str(p).startswith(str(FONTGEN_DIR.resolve())) or not p.exists():
            return self._json(404, {"ok": False, "error": "not found"})
        body = p.read_bytes(); ctype = "application/zip" if p.suffix == ".zip" else "font/ttf"
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Disposition", f'attachment; filename="{name}"'); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _zh_convert(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"ok": False, "error": "bad content-length"})
        if length <= 0 or length > 64 * 1024:
            return self._json(400, {"ok": False, "error": "empty or too large"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", ""))
            to = "t" if payload.get("to") == "t" else "s"
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad json: {e}"})
        try:
            from opencc import OpenCC
            cc = OpenCC("s2t" if to == "t" else "t2s")
            return self._json(200, {"ok": True, "text": cc.convert(text)})
        except Exception as e:
            return self._json(500, {"ok": False, "error": f"convert failed: {e}"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/zh-convert":
            return self._zh_convert()
        if path == "/api/fontgen/regen":
            return self._fontgen_regen()
        if path == "/api/fontgen/approve":
            return self._fontgen_approve()
        if path == "/api/fontgen/build":
            return self._fontgen_build()
        if path == "/api/fontgen/batch/start":
            return self._fontgen_batch_start()
        if path == "/api/fontgen/batch/stop":
            return self._fontgen_batch_stop()
        if path == "/api/ai-glyph":
            return self._ai_glyph()
        if path == "/api/ai-variants":
            return self._ai_variants()
        if path == "/api/save-glyph":
            return self._save_glyph()
        if path == "/api/font1/upload":
            return self._font1_upload()
        if path in ("/api/font1/label", "/api/font1/select", "/api/font1/delete"):
            return self._font1_update(path.rsplit("/", 1)[1])
        if path != "/upload":
            return self.send_error(404)

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"ok": False, "error": "bad content-length"})
        if length <= 0 or length > 50 * 1024 * 1024:
            return self._json(400, {"ok": False, "error": "empty or too large"})

        data = self.rfile.read(length)
        ensure_dirs()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        zip_path = UPLOAD_DIR / f"upload-{stamp}.zip"
        zip_path.write_bytes(data)

        saved = []
        skipped = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    m = GLYPH_NAME.match(info.filename)
                    if not m:
                        continue
                    char = m.group(1)
                    if len(char) != 1:
                        skipped.append(info.filename)
                        continue
                    raw = zf.read(info)
                    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                        skipped.append(info.filename)
                        continue
                    target = DATA_DIR / f"{char}.png"
                    target.write_bytes(raw)
                    saved.append(char)
        except zipfile.BadZipFile:
            return self._json(400, {"ok": False, "error": "not a zip"})

        global GLYPH_CACHE, SERVER_STATE
        GLYPH_CACHE = None
        SERVER_STATE = None
        print(f"[upload] {zip_path.name}: saved {len(saved)} glyphs -> {DATA_DIR}")
        return self._json(200, {
            "ok": True,
            "count": len(saved),
            "saved": saved,
            "skipped": skipped,
            "zip": str(zip_path),
            "dir": str(DATA_DIR),
        })


def main():
    ensure_dirs()
    try:
        Handler._fontgen_resume_jobs(None)
    except Exception as e:
        print(f"[fontgen] resume failed: {e}")
    def _prewarm():
        for fid in (0, 1):
            try:
                Handler._ensure_worker(None, fid)
                print(f"[ai] worker font{fid} prewarmed")
            except Exception as e:
                print(f"[ai] prewarm font{fid} failed: {e}")
    threading.Thread(target=_prewarm, daemon=True).start()
    print(f"serving {WEB_DIR} on 0.0.0.0:{PORT}")
    print(f"glyph target: {DATA_DIR}")
    print(f"upload archive: {UPLOAD_DIR}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
