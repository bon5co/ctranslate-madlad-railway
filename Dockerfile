# Translation API (419+ languages) on CTranslate2 serving Google MADLAD-400-3B-MT, LibreTranslate-compatible.
# Source: https://github.com/bon5co/ctranslate-madlad-railway
# CPU only, int8 weights baked in (2.95 GB), Bearer API_KEY, stateless (no volume needed).
# upstream: OpenNMT/CTranslate2 CTranslate2 4.9.0 (newest release, checked 2026-11-22) is pinned in requirements.txt
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip list 2>/dev/null | grep -iE "^(ctranslate2|sentencepiece|numpy|langcodes)"

# Weights: santhosh/madlad400-3b-ct2 is a CTranslate2 int8 conversion of google/madlad400-3b-mt
# (Apache-2.0 on both model cards). Pinned to one commit and verified by sha256, so a changed or
# replaced file fails the build instead of shipping. A download at boot would sit inside Railway's
# healthcheck window and need outbound network.
ARG HF=https://huggingface.co/santhosh/madlad400-3b-ct2/resolve/c32ad0cf118807ea6258d14be137547155842723
ADD --checksum=sha256:f3c87256a2c888100c179d7dcd7f41df17c767469546c59d32c7dde86c740a6b ${HF}/model.bin /models/madlad400-3b-ct2/model.bin
ADD --checksum=sha256:a428c51cd35517554523b3c6b6974a5928bc35e82b130869a543566a34a83b93 ${HF}/config.json /models/madlad400-3b-ct2/config.json
ADD --checksum=sha256:ef11ac9a22c7503492f56d48dce53be20e339b63605983e9f27d2cd0e0f3922c ${HF}/sentencepiece.model /models/madlad400-3b-ct2/sentencepiece.model
ADD --checksum=sha256:c327551ce3ca6efc7b437e11a267f79979893332dda8a1d146e2c950815193f8 ${HF}/shared_vocabulary.json /models/madlad400-3b-ct2/shared_vocabulary.json

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin ct2 \
    && chmod -R a+rX /models
WORKDIR /app
COPY app.py entrypoint.sh NOTICE LICENSE /app/
RUN chmod +x /app/entrypoint.sh
ENV MODEL_DIR=/models/madlad400-3b-ct2
USER 1000
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
