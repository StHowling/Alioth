#APP=${1:-'ffmpeg'}
#CODE=${2:-'m4'}

#echo ${APP},${CODE}

#################### This is an example on how to use auto archiving scripts  ###################

echo "run 1"
bash run-one.sh $1 $2
sleep 10

echo "run 2"
bash run-one.sh $1 $2
sleep 10

echo "run 3"
bash run-one.sh $1 $2
sleep 10

curl -X POST http://71.12.106.126:8090/v1/cmd -d '["clean_result","/home/fsp/czy/clean_result.py","'"$1"'","'"$2"'"]'
ip netns exec vpc-af694097-a56a-4817-84ab-e446ad9a6f33 python control.py "clean_result" $1 $2
curl -X DELETE http://71.12.106.126:8090/v1/cmd?command=clean_result


#
#echo "run 6"
#bash run-one.sh
#sleep 10
# 
# echo "run 7"
# bash run-one.sh
# sleep 10
# 
# echo "run 32"
# bash run-one.sh
# sleep 10
