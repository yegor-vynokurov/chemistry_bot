# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-container.txt ./requirements-container.txt
RUN --mount=type=secret,id=tutor_build_ca,target=/run/secrets/tutor-build-ca.crt,required=false \
    --mount=type=secret,id=avast_webshield_root,target=/run/secrets/legacy-avast-root.crt,required=false \
    if [ -s /run/secrets/tutor-build-ca.crt ] || [ -s /run/secrets/legacy-avast-root.crt ]; then \
        cat /etc/ssl/certs/ca-certificates.crt > /tmp/pip-ca-bundle.pem; \
        for certificate in /run/secrets/tutor-build-ca.crt /run/secrets/legacy-avast-root.crt; do \
            if [ -s "$certificate" ]; then printf '\n' >> /tmp/pip-ca-bundle.pem; cat "$certificate" >> /tmp/pip-ca-bundle.pem; fi; \
        done; \
        python -m pip install --no-cache-dir --cert /tmp/pip-ca-bundle.pem -r requirements-container.txt; \
        rm -f /tmp/pip-ca-bundle.pem; \
    else \
        python -m pip install --no-cache-dir -r requirements-container.txt; \
    fi

COPY src/chemistry_bot ./src/chemistry_bot

EXPOSE 8000

CMD ["python", "-m", "chemistry_bot.integrations.anythingllm_probe", "--host", "0.0.0.0", "--port", "8000"]
