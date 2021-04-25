FROM wynemo/python38:latest
COPY . flask_ml/
EXPOSE 1313
WORKDIR flask_ml/
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python service.py