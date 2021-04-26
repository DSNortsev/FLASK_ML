FROM wynemo/python38:latest
COPY . flask_ml/
EXPOSE 1313
WORKDIR flask_ml/
RUN pip install -r requirements.txt
CMD ["gunicorn", "service:app", "--config=config.py"]