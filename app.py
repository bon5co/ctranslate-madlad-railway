#!/usr/bin/env python3
"""Translation API for Railway: CTranslate2 serving MADLAD-400-3B, LibreTranslate-compatible.

CTranslate2 (MIT) runs Google's MADLAD-400-3B-MT (Apache-2.0) on CPU with int8 weights. Upstream
ships a Python library and no server. This file adds what a hosted deploy needs:

  1. Bearer API_KEY on every route except GET /health (LibreTranslate's JSON `api_key` field is
     accepted as well, so existing LibreTranslate clients work by only changing the URL).
  2. GET /health answers 503 until the model is loaded and one sentence has been translated, then
     200, one path segment deep and unauthenticated.
  3. The model weights are baked into the image (see Dockerfile): nothing downloads at boot.
  4. POST /translate and GET /languages with LibreTranslate's request and response shape.
     MADLAD does not need a source language (only a target prefix), so `source` may be "auto"
     or any code and is not used; there is no /detect route, because the model cannot detect.
  5. Requests run one at a time behind a lock; threads are sized from the container's cgroup
     quota, not the host's. Text is held in memory for one request and is never logged.
"""

import os
import re
import socket
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

API_KEY = os.environ.get("API_KEY") or ""
MODEL_DIR = os.environ.get("MODEL_DIR") or "/models/madlad400-3b-ct2"
THREADS = int(os.environ.get("CT2_CPU_THREADS") or os.cpu_count() or 1)
MAX_CHARS = int(os.environ.get("MAX_TEXT_CHARS") or 20_000)
MAX_BATCH = int(os.environ.get("MAX_BATCH") or 50)
CHUNK = int(os.environ.get("CHUNK_SEGMENTS") or 8)  # segments decoded together; bounds decoder memory
BEAM = int(os.environ.get("BEAM_SIZE") or 2)
SEGMENT_CHARS = int(os.environ.get("SEGMENT_CHARS") or 400)
PUBLIC_PATHS = {"/health"}
NOT_LANGUAGES = {"translate", "transliterate", "mass", "back_translated"}
SENTENCE_END = re.compile(r"(?<=[.!?。！？؟।])\s+")

app = FastAPI(
    title="Translation API (CTranslate2 + MADLAD-400-3B)",
    version="1.0.0",
    description="LibreTranslate-compatible translation on CPU with MADLAD-400-3B-MT (Apache-2.0) "
                "and CTranslate2 (MIT). Authorization: Bearer <API_KEY>.",
)

_state = {"error": None, "loaded_in": None, "ready": False}
_model: dict = {}
_lock = threading.Lock()


def _language_names(codes: list[str]) -> dict:
    try:
        import langcodes
    except Exception:  # noqa: BLE001 - names are a convenience, codes are the contract
        return {c: c for c in codes}
    names = {}
    for c in codes:
        try:
            names[c] = langcodes.Language.get(c.replace("_", "-")).display_name("en")
        except Exception:  # noqa: BLE001
            names[c] = c
        if names[c].lower().startswith("unknown") or "[" in names[c]:
            names[c] = c
    return names


def _load() -> None:
    t0 = time.time()
    try:
        import ctranslate2
        from sentencepiece import SentencePieceProcessor

        sp = SentencePieceProcessor()
        sp.load(os.path.join(MODEL_DIR, "sentencepiece.model"))
        codes = sorted(
            p[2:-1] for p in (sp.id_to_piece(i) for i in range(sp.get_piece_size()))
            if re.fullmatch(r"<2[a-z][A-Za-z0-9_-]*>", p) and p[2:-1] not in NOT_LANGUAGES)
        tr = ctranslate2.Translator(MODEL_DIR, device="cpu", inter_threads=1, intra_threads=THREADS,
                                    compute_type="int8")
        _model.update(sp=sp, tr=tr, codes=set(codes), names=_language_names(codes))
        _translate(["Warm-up sentence for the first request."], "de")
        _state["ready"] = True
        _state["loaded_in"] = round(time.time() - t0, 1)
        print(f"[ct2] ready languages={len(codes)} threads={THREADS} in {_state['loaded_in']}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported over /health, never fatal
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ct2] LOAD FAILED: {_state['error']}", flush=True)


def _segments(text: str) -> list[tuple[str, str]]:
    """Split into (separator-before, piece). Lines stay lines; long lines split at sentence ends."""
    out: list[tuple[str, str]] = []
    for n, line in enumerate(text.split("\n")):
        sep = "\n" if n else ""
        if len(line) <= SEGMENT_CHARS or not line.strip():
            out.append((sep, line))
            continue
        pieces, cur = [], ""
        for sent in SENTENCE_END.split(line):
            if cur and len(cur) + len(sent) + 1 > SEGMENT_CHARS:
                pieces.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}" if cur else sent
        pieces.append(cur)
        for k, p in enumerate(pieces):
            out.append((sep if k == 0 else " ", p))
    return out


def _translate(texts: list[str], target: str) -> list[str]:
    sp, tr = _model["sp"], _model["tr"]
    plans = [_segments(t) for t in texts]
    todo = [(i, j, p) for i, plan in enumerate(plans) for j, (_, p) in enumerate(plan) if p.strip()]
    outs = {}
    for k in range(0, len(todo), CHUNK):
        part = todo[k:k + CHUNK]
        batch = [sp.encode(f"<2{target}> {p.strip()}", out_type=str) for _, _, p in part]
        results = tr.translate_batch(batch, batch_type="tokens", max_batch_size=256, beam_size=BEAM,
                                     max_input_length=1024, max_decoding_length=512)
        for (i, j, _), res in zip(part, results):
            outs[(i, j)] = sp.decode(res.hypotheses[0])
    final = []
    for i, plan in enumerate(plans):
        buf = []
        for j, (sep, piece) in enumerate(plan):
            buf.append(sep + (outs[(i, j)] if (i, j) in outs else piece))
        final.append("".join(buf))
    return final


@app.middleware("http")
async def guards(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    ok = scheme.lower() == "bearer" and token == API_KEY
    if not ok and request.method == "POST":
        # LibreTranslate clients send the key in the JSON body. Read it only when no header is given.
        try:
            body = await request.json()
            ok = isinstance(body, dict) and body.get("api_key") == API_KEY and bool(API_KEY)
        except ValueError:
            ok = False
    if not ok:
        return JSONResponse(
            {"error": "Missing or invalid Authorization: Bearer <API_KEY>."},
            status_code=401, headers={"WWW-Authenticate": "Bearer"})
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BATCH * MAX_CHARS * 4 + 65536:
        return JSONResponse({"error": "request body is too large"}, status_code=413)
    return await call_next(request)


def _status_kb(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


@app.get("/health", include_in_schema=False)
def health():
    if not _state["ready"]:
        return JSONResponse({"status": "loading" if not _state["error"] else "failed",
                             "error": _state["error"]}, status_code=503)
    return {"status": "ok", "model": "madlad400-3b-mt (CTranslate2 int8)",
            "languages": len(_model["codes"]), "loaded_in_s": _state["loaded_in"],
            "rss_mb": _status_kb("VmRSS") // 1024, "peak_rss_mb": _status_kb("VmHWM") // 1024}


def _need_ready():
    if not _state["ready"]:
        raise HTTPException(503, _state["error"] or "model is still loading, retry in a few seconds")


@app.get("/languages")
def languages(targets: bool = False):
    """LibreTranslate shape. Every language translates to every other, so the `targets` list
    (454 codes per language, about 1.5 MB in all) is only attached when asked for: ?targets=true."""
    _need_ready()
    codes = sorted(_model["codes"])
    return [{"code": c, "name": _model["names"].get(c, c), **({"targets": codes} if targets else {})}
            for c in codes]


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{
        "id": "madlad400-3b-mt-ct2-int8", "object": "model", "owned_by": "google",
        "license": "apache-2.0", "max_text_chars": MAX_CHARS, "max_batch": MAX_BATCH}]}


@app.post("/translate")
async def translate(request: Request):
    _need_ready()
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(415, "send application/json")
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(422, "body is not valid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(422, "body must be a JSON object")
    q = body.get("q")
    batch = isinstance(q, list)
    texts = q if batch else [q]
    if not texts or not all(isinstance(t, str) for t in texts):
        raise HTTPException(422, "`q` must be a string or a non-empty list of strings")
    if len(texts) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} texts per request")
    if any(len(t) > MAX_CHARS for t in texts):
        raise HTTPException(413, f"a text is longer than the {MAX_CHARS} character limit")
    if body.get("format", "text") != "text":
        raise HTTPException(422, "only format \"text\" is supported, markup would be translated as text")
    target = body.get("target")
    if not isinstance(target, str) or target not in _model["codes"]:
        raise HTTPException(422, f"unsupported target language {target!r}, see GET /languages")
    t0 = time.time()
    with _lock:
        try:
            outs = _translate(texts, target)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc
    took = time.time() - t0
    payload = {"translatedText": outs if batch else outs[0]}
    return JSONResponse(payload, headers={"X-Processing-Seconds": f"{took:.3f}"})


def main() -> int:
    if not API_KEY:
        print("[ct2] FATAL: API_KEY is empty; refusing to publish the API to anyone who finds the URL.",
              file=sys.stderr, flush=True)
        return 1
    threading.Thread(target=_load, name="load", daemon=True).start()
    # Railway injects PORT and its healthcheck dials that injected port. asyncio sets
    # IPV6_V6ONLY on a "::" listener, so one socket per family: IPv4 for the edge and
    # healthcheck, IPv6 for the private network (<service>.railway.internal).
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[ct2] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        return 1
    config = uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "warning"))
    uvicorn.Server(config).run(sockets=socks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
