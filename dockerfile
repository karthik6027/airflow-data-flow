FROM apache/airflow:latest

USER root
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

USER airflow

RUN pip install numpy
RUN pip install pandas 
RUN pip install great_expectations


#RUN pip install -r requirements.txt



