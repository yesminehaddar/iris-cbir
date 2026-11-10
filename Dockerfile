FROM python:3.12-slim

WORKDIR /srv

# Wheels exist for every dependency, so no build toolchain is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# The image base and the feature matrices are bind-mounted, not copied: the
# thumbnail base is 800k files and the index is rebuilt rather than shipped.
ENV FLICKR_DIR=/srv/thumbnails \
    FLICKR_FEATURES_DIR=/srv/data/features/flickr1m

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
