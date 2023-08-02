FROM python:latest
RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/
COPY . /usr/src/app/
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
CMD ["python", "ocr.py"]