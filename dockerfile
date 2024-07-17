
#Using python
FROM python:3.11.7

RUN mkdir dash-app
WORKDIR /dash-app

# Using Layered approach for the installation of requirements
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


CMD ["gunicorn","--reload",  "-b", "0.0.0.0:8080" ,"app:server"]
