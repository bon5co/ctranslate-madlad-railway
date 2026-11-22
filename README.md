# ctranslate-madlad-railway

Translation API on [CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT) serving Google's [MADLAD-400-3B-MT](https://huggingface.co/google/madlad400-3b-mt) (Apache-2.0, 400+ languages) on CPU, for Railway. LibreTranslate-compatible request and response shape.

- `POST /translate` `{"q": "text" | ["a", "b"], "source": "auto", "target": "de"}` -> `{"translatedText": ...}`. `source` is accepted and not used (MADLAD takes only a target); `format` must be `text`.
- `GET /languages` -> `[{"code", "name"}]` (454 target codes; `?targets=true` adds the LibreTranslate `targets` list). `GET /v1/models`, `GET /health` (503 until the model is loaded and warm, then 200, with `peak_rss_mb`).
- Auth: `Authorization: Bearer $API_KEY`, or LibreTranslate's `api_key` field in the JSON body. `/health` is open.
- Env: `API_KEY` (required), `BEAM_SIZE` (2), `MAX_TEXT_CHARS` (50000), `MAX_BATCH` (50), `SEGMENT_CHARS` (400), `CT2_CPU_THREADS` (from the cgroup quota).

Weights (2.95 GB int8, [santhosh/madlad400-3b-ct2](https://huggingface.co/santhosh/madlad400-3b-ct2) at a pinned commit, sha256 verified at build) are baked into the image. Text is held in memory for one request and never logged or stored. Wrapper code is Apache-2.0 (`LICENSE`); see `NOTICE`.

Image: `ghcr.io/bon5co/ctranslate-madlad-railway`.
