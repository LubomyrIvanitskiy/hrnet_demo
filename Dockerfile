FROM hrnet
COPY app.py app.py
CMD ["python", "app.py", "serve"]
