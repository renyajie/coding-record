a brief records about some useful operations in docker.

* docker login: 登录

* docker --version: 查看docker版本

* dokcer build -t friendlyhello .: 用当前文件夹的Dockerfile创建friendlyhello镜像(image)

* docker images: 查看本地的docker镜像
* docker image ls -a: 上同
* docker image rm <image id>: 删除特定的image
* docker image rm $(docker image ls -a -q): 删除全部的image

* docker run -p 4000:80 friendlyhello: 运行friendlyhello，并将本机的4000端口和container的80进行映射
* docker run -d -p 4000:80 friendlyhello: 和上面相同，但以分离模式运行

* docker container ls: 查看当前运行中的容器
* docker container ls -a: 查看所有容器
* docker container ls -q: 查看全部容器的ID

* docker container stop 1fa4ab2cf395: 停止运行ID为1fa4ab2cf395的容器
* docker container kill 1fa4ab2cf395: 强制关闭ID为1fa4ab2cf395的容器
* docker container rm <hash>: 从本机移除指定ID的容器
* docker container rm $(docker container ls -a -q): 移除所有的容器

* docker tag image username/repository:tag: 为镜像打上标签
    docker tag friendlyhello john/get-started:part2

* docker push username/repository:tag: 上传打过标签的镜像到目的仓库
* docker run username/repository:tag: 运行仓库中的镜像
    docker run hello-world: 运行hello world

* docker stack ls: 列出运行中的app
* docker stack deploy -c <composefile> <appname>: 运行Compse文件，并以appname进行命名
    docker stack deploy -c docker-compose.yml getstartedlab
* docker stack rm <appname>: 从stack中拆除某个app
    docker stack rm getstartedlab

* docker service ls: 查看app中的运行中的service
* docker service ps <service>: 查看某个service中的task(运行中的容器)
    docker service ps getstartedlab_web

* docker swarm init: 初始化一个swarm
* docker swarm leave --force: 让一个node离开swarm

* docker-machine ls: 查看虚拟机
* docker-machine create -d hyperv --hyperv-virtual-switch "myswitch" myvm1: (win10,hyperv)创建myvm1虚拟机
* docker-machine ssh myvm1 "docker swarm init --advertise-addr <myvm1 ip>": 让虚拟机myvm1初始化swarm，并成为manager
* docker-machine ssh myvm2 "docker swarm join --token <token> <ip>:2377": 虚拟机myvm2加入swarm
* docker-machine ssh myvm1 "docker node ls": 查看swarm中的节点
* docker-machine start <machine-name>: 启动指定的虚拟机
* docker-machine stop <machine-name>: 停止指定的虚拟机
* docker-machine stop $(docker-machine ls -q): 停止所有的虚拟机
* docker-machine rm $(docker-machine ls -q): 删除所有的虚拟机

* docker-machine env myvm1
* ~(Run the given command to configure your shell to talk to myvm1): 使用myvm1进行shell指令交互，同时可访问本地文件
