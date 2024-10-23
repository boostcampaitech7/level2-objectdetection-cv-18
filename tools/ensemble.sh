#!/bin/zsh
python3 ensemble.py --name nms 
python3 ensemble.py --name non_maximum_weighted 
python3 ensemble.py --name weighted_boxes_fusion