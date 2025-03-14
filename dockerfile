FROM apache/airflow:latest

USER root
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

USER airflow

RUN pip install numpy
RUN pip install pandas 
RUN pip install great_expectations==0.18.19
RUN pip install seaborn
RUN pip install missingno
RUN pip install scikit-learn
RUN pip install plotly




#RUN pip install -r requirements.txt



