# docker run -it iris_azure sh - enter image to ls files for instance
# docker rm -f $(docker ps -aq) - close all running containers
# docker run -p 8000:8000 -t -i iris_azure - run image

FROM python:3.9
COPY ./requirements.txt /docker/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /docker/requirements.txt
COPY ./ /docker
WORKDIR /docker
EXPOSE 8000:8000
CMD ["gunicorn", "-w 4", "app:app", "-b 0.0.0.0:8000", "-k uvicorn.workers.UvicornWorker"]