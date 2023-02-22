CURRENT_DIR=$(cd $(dirname $0); pwd)
python3 -W ignore ${CURRENT_DIR}/inference.py --bbox -times 1000
