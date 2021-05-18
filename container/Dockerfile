FROM ubuntu:18.04
#FROM  python:3.8

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         python3-pip \
         python3-setuptools \
         python3-dev \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip --no-cache-dir install boto3 wheel numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas flask
RUN pip --no-cache-dir install gevent gunicorn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY coupon_recommendation_engine /opt/program
WORKDIR /opt/program
