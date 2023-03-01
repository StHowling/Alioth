addr=0.0.0.0
if [ $1 = '3' ]; then
    addr=192.168.0.128
elif [ $1 = '5' ]; then
    addr=192.168.0.110
elif [ $1 = '7' ]; then
    addr=192.168.0.245
elif [ $1 = '8' ]; then
    addr=192.168.0.170
elif [ $1 = '9' ]; then
    addr=192.168.0.233
elif [ $1 = '10' ]; then
    addr=192.168.0.248
elif [ $1 = '11' ]; then
    addr=192.168.0.94
elif [ $1 = '12' ]; then
    addr=192.168.0.220
elif [ $1 = '13' ]; then
    addr=192.168.0.11
elif [ $1 = 'c2' ]; then
    addr=192.168.0.44
else
    echo align_system_time.sh domID
    exit 1
fi

printf -v date '%(%Y%m%d%H%M%S)T'
year=${date:0:4}
month=${date:4:2}
day=${date:6:2}
time="$year-$month-$day"
curl -X POST http://$addr:8081/v1/cmd -d '["align","/root/align_system_time.sh","'"$time"'"]'
curl -X DELETE http://$addr:8081/v1/cmd?command=align

printf -v date '%(%Y%m%d%H%M%S)T'
hour=${date:8:2}
minute=${date:10:2}
second=${date:12:2}
time="$hour:$minute:$second"

#echo $time
curl -X POST http://$addr:8081/v1/cmd -d '["align","/root/align_system_time.sh","'"$time"'"]'
curl -X DELETE http://$addr:8081/v1/cmd?command=align
