
Deploy your simple pose estimation service on service

The project is based on https://towardsdatascience.com/10-minutes-to-deploying-a-deep-learning-model-on-google-cloud-platform-13fa56a266ee tutorial and the
https://github.com/stefanopini/simple-HRNet repository

To deploy the project on your Linux (tested on Ububntu) machine run the following commands:

1. Install docker. Follow https://docs.docker.com/engine/install/ubuntu/ or:
1.1 chmod 755 install_docker.sh
1.2 sudo sh install_docker.sh
2. git clone https://github.com/LubomyrIvanitskiy/hrnet_demo.git
3. cd hrnet_demo
4. chmod 755 setup_hrnet.sh
5. sh setup_hrnet.sh
6. sudo docker image build -f Dockerfile-base -t base requirements
7. sudo docker image build -t app:latest . #Do not miss the dot in the end
8. sudo docker run -d -p 80:8008 app:latest
