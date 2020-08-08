#Use python as base image
FROM ubuntu:16.04

#WORKDIR /app
#ADD . /app/
#RUN mkdir /lab

#WORKDIR /lab
#ADD . /lab/


# Install system packages (python 3.5)
RUN apt-get update && \
	apt-get remove -y \
	x264 libx264-dev && \
	apt-get install -y \
	build-essential \
	checkinstall \
	cmake \
	pkg-config \
	libjpeg8-dev \
	libjasper-dev \
	libpng12-dev \
	libtiff5-dev \
	libtiff-dev \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	libdc1394-22-dev \
	libxine2-dev \
	libv4l-dev

RUN apt-get install -y \
	libgstreamer0.10-dev \
	libgstreamer-plugins-base0.10-dev \
	libgtk2.0-dev \
	libtbb-dev \
	qt5-default \
	libatlas-base-dev \
	libfaac-dev \
	libmp3lame-dev \
	libtheora-dev \
	libvorbis-dev \
	libxvidcore-dev \
	libopencore-amrnb-dev \
	libopencore-amrwb-dev \
	libavresample-dev \
	x264 \
	v4l-utils \
	libprotobuf-dev \
	protobuf-compiler \
	libgoogle-glog-dev \
	libgflags-dev \
	libgphoto2-dev \
	libeigen3-dev \
	libhdf5-dev \
	doxygen


RUN apt-get install -y \
	python-dev \
	python-pip \
	python3-dev \
	python3-pip


RUN  pip install -U pip

RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip3 install opencv-contrib-python


RUN pip3 install numpy==1.16.5
RUN pip3 install pickleshare==0.7.5
RUN pip3 install cloudpickle==1.2.2
RUN pip3 install scikit-learn==0.21.3
RUN pip3 install scikit-learn==0.21.3
RUN pip3 install Flask==0.11.1
RUN pip3 install imutils==0.5.3
RUN pip3 install json5==0.8.5
RUN pip3 install gunicorn==19.9.0
RUN pip3 install Werkzeug==0.15.5

# Setting up working directory 
#RUN mkdir /lab

ADD  . /

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)


#Open port 5000
#EXPOSE 5000

#Set environment variable
#ENV NAME OpentoAll

#Run python program
#CMD ["python3","app.py"]


# ssh
ENV SSH_PASSWD "root:Docker!"
RUN apt-get update \
        && apt-get install -y --no-install-recommends dialog \
        && apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && echo "$SSH_PASSWD" | chpasswd 

COPY sshd_config /etc/ssh/
COPY init.sh /usr/local/bin/
    
RUN chmod u+x /usr/local/bin/init.sh
EXPOSE 8000 2222

#service SSH start
#CMD ["python", "/code/manage.py", "runserver", "0.0.0.0:8000"]
ENTRYPOINT ["init.sh"]






