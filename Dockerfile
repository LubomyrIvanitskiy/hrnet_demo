FROM base
COPY hrnet hrnet
WORKDIR hrnet
COPY weights weights
COPY app.py app.py
CMD ["python", "app.py", "serve"]
