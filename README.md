
Deploy your simple pose estimation service on Google Cloud

The project is based on https://towardsdatascience.com/10-minutes-to-deploying-a-deep-learning-model-on-google-cloud-platform-13fa56a266ee tutorial and the
https://github.com/stefanopini/simple-HRNet repository

To deploy the project on your Linux (tested on Ububntu) machine run the following commands:

<li>1. git clone https://github.com/LubomyrIvanitskiy/hrnet_demo.git
<li>2. cd hrnet_demo
<li>3. Install docker. Follow https://docs.docker.com/engine/install/ubuntu/ or:
<li>3.1 chmod 755 install_docker.sh
<li>3.2 sudo sh install_docker.sh
<li>4. chmod 755 setup_hrnet.sh
<li>5. sh setup_hrnet.sh
<li>6. sudo docker image build -f Dockerfile-base -t base requirements
<li>7. sudo docker image build -t app:latest . #Do not miss the dot in the end
<li>8. sudo docker run -d -p 80:8008 app:latest

![alt text](https://raw.githubusercontent.com/LubomyrIvanitskiy/hrnet_demo/main/hrnet_demo.png)
