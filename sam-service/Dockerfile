FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 wget git \
    && apt-get clean

RUN mkdir -p /data/sam && \
    wget -q -O /data/sam/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sam_service.py .

ENV SAM_CHECKPOINT=/data/sam/sam_vit_b_01ec64.pth
ENV SAM_MEMORY=/data/sam_memory.json
ENV PORT=8001

CMD ["python", "sam_service.py"]
