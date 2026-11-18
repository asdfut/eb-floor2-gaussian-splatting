#!/usr/bin/env bash
# run FAST-LIO (Mid-360 + its IMU) offline on one bag; writes TUM poses (IMU/body frame in the initial frame)
# usage: run_fastlio.sh <bag_dir> <out.tum> [rate]
BAG=$1; OUT=$2; RATE=${3:-2.0}
HERE=$(cd $(dirname $0) && pwd)
source /opt/ros/jazzy/setup.bash; source $HERE/../install/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
rm -f $OUT
# start the node binary directly (not via `ros2 run`, whose wrapper does not forward signals to the node)
$HERE/../install/fast_lio/lib/fast_lio/fastlio_mapping --ros-args --params-file $HERE/mid360_offline.yaml > ${OUT%.tum}_fastlio.log 2>&1 &
P1=$!
/usr/bin/python3 $HERE/odom_logger.py $OUT > /dev/null 2>&1 &
P2=$!
sleep 3
ros2 bag play $BAG --topics /livox/lidar /livox/imu -r $RATE > ${OUT%.tum}_play.log 2>&1
sleep 3
kill -INT $P2 $P1 2>/dev/null
for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $P1 2>/dev/null || break; sleep 1; done
kill -9 $P1 $P2 2>/dev/null; wait $P1 $P2 2>/dev/null
N=$(wc -l < $OUT); D=$(awk 'NR==1{a=$1} END{print $1-a}' $OUT)
echo "$(basename $BAG): $N poses over ${D}s ($(awk -v n=$N -v d=$D 'BEGIN{printf "%.1f", n/d}') Hz) -> $OUT"
