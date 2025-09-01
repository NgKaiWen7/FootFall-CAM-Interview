FROM archlinux:latest
RUN pacman -Syu --noconfirm python python-pip git base-devel cuda cudnn nvidia
RUN rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
RUN pip3 install ultralytics opencv-python
COPY . .
RUN chmod +x run_pipeline.sh
CMD ["./run_pipeline.sh"]
