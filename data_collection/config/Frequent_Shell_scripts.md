# Frequent shell scripts in data collection
**Note: copy the commands in the preview rather than raw .md**

## Set up cmdserver
You can run multiple cmdserver at the same time on the same machine, as long as their listening ports are different.

| Machine Type | Shell Command | Explanation |
| ------------ | ------------- | ----------- |
| Client VM | nohup ./cmdserver -address 192.168.0.180 -port 8090 -path python & | nohup & : run at backstage, continue even this terminal exits; address: local IP of client VM; port: default; path: program to be remotely called, for client vm should be python to execute load_stress_runner.py |
| App VM (Batch jobs) | nohup ./cmdserver -address 192.168.0.xxx  -path /bin/bash -port 8081 & | xxx: specific local IP; port: app vms all use ~8080~ 8081 *8080 should be avoided since it is http-alt*; path: batch jobs are usually called via shell, with corresponding run-xxx.sh |
| Workload PM | nohup ./cmdserver -address 71.12.110.25 -port 8090 -path python & | Workload PM also relies on two python scripts to collect micro and macro metrics |
| Stress VM |  nohup ./cmdserver -address 192.168.0.xxx -port 8082 -path /root/stress-ng & | Note: path may also be python (port 8090) since code provided by Prof. Chen uses python; different programs being remotely called should use different port |

## Set up applications

Without specification, use `docker kill $containerID` and `docker start $containerID` to stop and start an application.

The following scripts are, by default, for the first-time configurations.

### **etcd**
> $ scp root@192.168.0.128:/root/docker-image/etcd.tar /root/docker-image/   
> $ docker load --input /root/docker-image/etcd.tar   
> $ docker run -d \\   
> -p 2379:2379 \\   
> --name etcd \\    
> etcd:latest \\  
> /home/etcd \\  
> --listen-client-urls http://0.0.0.0:2379 \\  
> --advertise-client-urls http://$ClientIP:2379 

Note that after several times of 2-hour test, the DB server may run out of space and raise exceptions. No need to setup auto compaction parameters, just `docker rm` it and restart a new one. 

### **redis**

> $ scp root@192.168.0.128:/root/docker-image/redis.tar /root/docker-image/   
> $ docker load --input /root/docker-image/redis.tar    
> $ docker run -d \\   
> -p 6379:6379 \\   
> --name redis \\    
> redis:latest \\  
> redis-server 

**Caution** When dealing with YCSB-benchmarks, remember to create tables specially for YCSB and to load data into the specific dataset before test. Otherwise thousands of NPE and READ-FAILED awaits you. MongoDB is the only app in YCSB that does not require special setting (i.e., creating special tables/keyspaces for YCSB before load).

### **mongodb**

Sever:

> $ docker load --input /root/docker-image/mongodb.tar   
> $ docker run -d \\   
> -p 27017:27017 \\   
> --name mongodb \\    
> mongodb:latest

Client:

First modify the `host/addr/url` keyspace in workload file.

> $ /root/software/ycsb/bin/ycsb load mongodb -s -P /root/mongodb-workload > mongoload.log

You can change the workload file parameters to adapt to our needs. The most important options are :`readallfields`, `readproportion`, `updateproportion`, `scanproportion`, `insertproportion`, `requestdistribution`. To mimic real-world workloads, keep `readproportion` high since data-serving latency-critical services receives read queries the most.

You can also test the mongodb-async working mode. Subtitute "mongodb" with "mongodb-async" and change the workload configurations accordingly.

### **cassandra** 
  
Server:

> $ scp root@192.168.0.128:/root/docker-image/cassandra.tar /root/docker-image/   
> $ docker load --input /root/docker-image/cassandra.tar   
> $ docker run -d \\   
> -p 9042:9042 \\   
> --name cassandra \\    
> cassandra:latest    
> $ docker exec -it cassandra /bin/bash   
> \# cqlsh   

```
cqlsh> create keyspace ycsb
    WITH REPLICATION = {'class' : 'SimpleStrategy', 'replication_factor': 3 };
cqlsh> USE ycsb;
cqlsh> create table usertable (
    y_id varchar primary key,
    field0 varchar,
    field1 varchar,
    field2 varchar,
    field3 varchar,
    field4 varchar,
    field5 varchar,
    field6 varchar,
    field7 varchar,
    field8 varchar,
    field9 varchar);
```

Client:

First modify the `host/addr/url` keyspace in workload file.

> $ /root/software/ycsb/bin/ycsb load cassandra-cql -s -P /root/cassandra-workload > cassandraload.log

### **habse**

Server:

> $ docker run -d \\   
> -p 2181:2181 \\   
> -p 16020:16020 \\   
> -p 16010:16010 \\   
> -p 16000:16000 \\   
> -p 9090:9090 \\   
> -p 9095:9095 \\   
> --name hbase \\    
> registry-cbu.huawei.com/huawei-vod/hbase:latest    
> $ docker exec -it hbase /bin/sh   
> \# hbase shell

```
hbase(main):001:0> n_splits = 100 # HBase recommends (10 * number of regionservers)
hbase(main):002:0> create 'usertable', 'family', {SPLITS => (1..n_splits).map {|i| "user#{1000+i*(9999-1000)/n_splits}"}}
```

Client:

Modify IP in `/root/software/hbase/conf/hbase-sites.xml`.

Add IP of server and **docker container ID** as host name in /etc/hosts.

> $ /root/software/ycsb-hbase2-binding/bin/ycsb load hbase2 -s -cp /root/software/hbase/conf -P /root/hbase-workload > hbaseload.log

### **rabbitmq**

> $ docker load --input /root/docker-image/rabbitmq.tar   
> $ docker tag xxxx rabbitmq //df80 by default, check via docker image list  
> $ docker run -d \\   
> -p 5672:5672 \\   
> --name rabbitmq \\    
> rabbitmq:latest



### **kafka**

After a frustrating day trying to pack it into container and fixing linstening port issues, it is sad to conclude that current version of kafka & its test benchmark cannot be deployed susccessfully in our docker environment. We resort to use shell scripts, which also seems to be convenient enough. The prepartions part is for first-time configuration, and in practice just use the last two commands.

Preparations    
Copy jdk and kafka directory under /root/software from instance-156.
> vim /etc/profile    
> export JAVA_HOME=/root/software/jdk1.8.0_212 //append at bottom, save, quit    
> source /etc/profile    
> vim `$kafkapath`/config/server.properties    
> listeners=PLAINTEXT://`loacl IP`:9092 //change line 31, save, quit

Normal launch
> `$kafkapath`/bin/zookeeper-server-start.sh -daemon `$kafkapath`/config/zookeeper.properties    
> `$kafkapath`/bin/kafka-server-start.sh -daemon `$kafkapath`/config/server.properties

Or for convenience 
> /root/software/start-kafka.sh

Stopping serving
> kill -s 9 \`ps -aux | grep kafka | grep -v 'grep' | awk '{print $2}'`

 
### **stress-ng**
Just copy the binary file and put it at `/root`. Remember to setup the cmdserver for stress-ng.

### **nginx/httpd**
Easy to configure. Maybe need to add some html pages into the container.
> docker load --input /root/docker-image/nginx.tar   
> docker run -d -p 80:80 -p 443:443 --name nginx nginx

> docker load --input /root/docker-image/apache-server.tar   
> docker run -d -p 80:80 -p 443:443 --name apache-httpd httpd

To view html pages in Edge/Chrome: remember to configure 71.12.* IP range as no proxy in system settings.

### **mysql**
~~Still a easy one.~~ Not as easy as it seems. On server side, `apt-get install mysql-server`. Configure root password as *QoSTeam*. To stop and start, use
> systemctl stop mysql   
> systemctl start mysql

Login to mysql shell with root user and create special user and database for sysbench test.
> mysql -p   
> \> create database sbtest;   
> \> create user 'sysbench'@'%' identified by 'sysbench';    
> \> grant all privileges on sbtest.* to 'sysbench'@'%' identified by 'sysbench';    
> \> flush privileges;

The next step is important. Change the binding address setting to allow remote login to mysql server.
> vim /etc/mysql/mysql.conf.d/mysqld.cnf
```
bind_address=0.0.0.0 
```

The old image used for creating VM has some conflicting configurations and causes installation failure. To fix, remove them completely and reinstall.
> apt-get remove --purge mysql-server mysql-client mysql-common    
> apt-get autoremove    
> apt-get autoclean    
> apt-get install mysql-server

There may not be a prompt guiding to set the password for root during this reinstallation. To fix, access the temporary name and password to login:
> cat /etc/mysql/debian.cnf    
> mysql -u debian-sys-maint -p    
> \> use mysql;   
> \> update mysql.user set authentication_string=password('QoSTeam') where user='root' and Host ='localhost';    
> \> update user set plugin="mysql_native_password";     
> \> flush privileges;    
> systemctl restart mysql

On client side, copy the binary file `/root/sysbench` on Client-1. Then copy the missing `.lib.so` files to corresponding path:
> scp root@192.168.0.180:/usr/lib/x86_64-linux-gnu/libmysqlclient.so.20 /usr/lib/x86_64-linux-gnu/   
> scp root@192.168.0.180:/usr/lib/x86_64-linux-gnu/libmysqlclient.so.20.3.18 /usr/lib/x86_64-linux-gnu/

Just as the ycsb benchmark, you also need to load data into the test database before running the test. This only needs to be done once (and takes effect even if we shut down mysql and restart), but for convenience I just write the code into load_stress_runner.py and make it load data everytime before running the test.
> ./sysbench --db-driver=mysql --mysql-host=192.168.0.233 --mysql-port=3306 --mysql-user=sysbench --mysql-password=sysbench --test=oltp prepare

The response should be like 
```
sysbench 0.4.12:  multi-threaded system evaluation benchmark

Creating table 'sbtest'...
Creating 10000 records in table 'sbtest'...

```

Multiple same loads will raise error like 
```
ALERT: failed to execute MySQL query: `CREATE TABLE sbtest (id INTEGER UNSIGNED NOT NULL AUTO_INCREMENT, k integer UNSIGNED DEFAULT '0' NOT NULL, c char(120) DEFAULT '' NOT NULL, pad char(60) DEFAULT '' NOT NULL, PRIMARY KEY (id) )  /*! ENGINE=innodb */`:
ALERT: Error 1050 Table 'sbtest' already exists
FATAL: failed to create test table
```

But does no harm to the test program. You can use advanced settings like `--oltp-test-mode`, `--oltp-tables-count` and `--oltp-table-size` to adjust the test.

### **LAMP/LNMP**
Namely, linux, apache web server/nginx, mysql/mariadb, and php. This is a multi-tier web service architecture. A typical example application based on it is e-commercial website. All components can be installed via apt-get and controlled via systemctl. Some additional packages for php may be needed.
> apt-get install apache2 mysql-server php libapache2-mod-php php-mysql php-bz2 php-curl php-gd     
> systemctl stop php7.0-fpm

A few configurations are needed for this environment, based on the sepecific application. For example, you may need to adjust what kind of web page the apache web server choose to search and show first in a directory when receiving a request from clients. For the e-comm website ab, move `index.php` to the first place.
> vim /etc/apache2/mods-enabled/dir.conf   
> \<IfModule mod_dir.c\>   
> DirectoryIndex index.html index.cgi index.pl index.php index.xhtml index.htm   
> \</IfModule\>

You may also need to create a dedicated database and corresponding account for each app. For test covenience and simplicity, allow the apps to use `root`.
> $ mysql -p   
> create database appname;   
> create user 'xxx'@'localhost' identified by passwd;   
> grant all privileges on appname.* to 'xxx'@'localhost';   
> flush privileges;

For the website deployment, it depends on the realization. Some websites are installed and configured via browser's first access (i.e. use Windows PC to access http://`Public IP` in Edge/Chrome). Just follow the instructions on the pages and you are done. For some you just need to extract files and copy them to the web sever working directory (by default `/var/www/html` for apache2). 

Be sure to put the index, install pages and lib, source directories directly under the working directory, not packed in a subdirectory. To change the working directory, for apache2 modify the following configuration:
> vim /etc/apache2/sites-enabled/000-default.conf   

Change `DocumentRoot` as you need. After modification, be sure to restart `apache2` service. To stop the application, just stop the AMP services, while the website files are maintained.

Note that apache web server is actually the same thing as httpd for versions above 2. In Ubuntu/Debian, they are known and managed as apache2; in RHEL/CentOS, they are known and managed as httpd.

The nginx and mariadb configurations are currently not provided since no app use them in our current scope. However they are very similar to apache2 and mysql.

## **Memcached**
For server, use `apt-get install memcached` to install. To start, use
> memcached -d -t 4 -m 256 -c 1024 -u memcache -l $LocalIP -p 11211

where `-d` means to start a daemon, `-t` means the number of worker threads, which is by default 4, `-m` is the maximum memory allocated in MB, which is by default 64, `-c` is the maximum number of connections to hold concurrently, which is by default 1024, `-l` is the IP address, `-p` is the port to start listening, which is by default 11211.

For client, use VMs of large flavor. Get the code of [mutilate](https://github.com/leverich/mutilate "mutilate") or mutated (haven't tried yet) from github. Unzip the file into a proper path, `cd` into it.
> apt-get install scons libevent-dev gengetopt libzmq-dev    
> scons

Before starting to put requests, we should load the server with entries. An example used by the Meta-LLCC experiment is as follows:
> mutilate -s $serverIP -K 30 -V 256 -r 400000 --loadonly    
where `-K` denotes the size of key in KB, `-V` denotes the size of value, `-r` denotes total number of records to load, `--loadonly` means to only put the records and the quit, not performing latency test. These values are by default 30, 256, 10000. We do not recommend to use master-agent mode of mutilate, since by practice it leads to worse tail latency. Instead, use serveral clients separately to reach the max QPS.

## Manage VMs via virsh
Normally, just use `virsh suspend domID` and `virsh resume domID`. This keeps their states in the memory of PM, and takes into effect immediately. This way the memory usage on the PM cannot be freed but all other resources (CPU, disk, net) can be freed for alternative use. A major concern is that when restored, the system time shows a giant lag and needs calibration. (This is vital as we use UNIX tiemstamp to merge the distributed micro-architecture and performance data.)

For time alignment, run the following command on Client-1:
> /root/align_system_time.sh `domID`

Stopping some VM and turning it into "inactive" state is not recommended. But in case we need it some time, here we describe two methods of save-stop-restore:
> virsh managedsave domain --bypass-cache --running | --paused | --verbose    
> virsh start domain

> virsh save domain [--bypass-cache] domain file [--xml string] [--running] [--paused] [--verbose]    
> virsh restore domain file

The result of managedsave is identical to the result of save. They use the same QEMU code internally, the only difference being that in one
case you decide the filename and in the other case libvirt decides
the filename.

We also intended to use `virsh emulatorpin domID [range] --live` to query affinity and pin the domain to specific cpus ~~and `virsh emulatorpin domID vcpuID cpuID --live` to pin some vcpu in 1-vs-1 manner. But the later command proved to be not working. Also there is no pinning command in `virsh` for *iothread2\**.~~ We resort to load overall config from a json file and write the values directly into `.../cgroup/.../cpuset.cpus` via python scripts.

The reason why `virsh emulatorpin domID vcpuID cpuID --live` does not work is that it does not comply to the command format of our virsh version. It should be `virsh vcpupin [--domain] domID --vcpu vcpuID --cpulist cpulist [--live]`. `emulatorpin` cannot be used to pin vcpus, and emulator threads adn vcpu threads are essentially different. The former handle interrupts and emulates other hardwares used by VM except from CPU, memory, i.e. everything but vcpu. Besides, note that `--vcpu` option accepts only a single `vcpuID`, `cpulist` can use dash to assign consequent pcpu and comma to assign separated pcpu, e.g. `8,10-12`, and `--vcpu`, `--cpulist` can not be ommited for short.

For *iothread*, there are also a set of commands. *iothread* is usually thought as a "new" (compared to very classical way of qemu virtualization) feature to fully leverage modern multi-core processors, preventing syncronized I/O block-ups to vcpu worker threads previously handled by the main qemu process. It act as a thread pull: whenever inside the VM calls for I/O, the requests are shifted and loaded on *iothreads*, and therefore supports asyncronized I/O. It is recommended to have 1~2 *iothreads* for each core. You can use `virsh iothreadadd threadID` and `virsh iothreaddel threadID` to increase or decrease the scale of *iothreads*. You can use `virsh iothreadpin [--domain] domID [--iothread] threadID [--cpulist] cpulist` to pin a sepcific iothread to some pcpu. Or you can use `virsh edit` to directly modify the xml description file, where there is a section sepcifying the configurations on *iothreads*. 

`virsh` is essentially a shell tool for `qemu-kvm`, which relies on `libvirt` to execute these commands and manage virtual machines. Concerning core and thread pinning, in the end it goes to `cgroup`, where `libvirt` just writes cpusets info into the special file system `/sys/fs/cgroup/cpuset/...`. This is the natural way of using `cgroup` to isolate programs in the user space, since `cgroup` does not have any cmd tools or advanced APIs. In the sense of VM isolation, `virsh xxxpin` can be viewed as an advanced API of `cgroup`. It does exactly the same thing as directly writing values into the cpuset configurations.

## Allocate VMs to specific cpus
Prepare the overall configuration .json file in Workloadhost:/home/fsp/czy/load_flavor. Then just `python load_vm_flavor.py`.