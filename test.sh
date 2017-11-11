#!/bin/sh

ipython utils.py
ipython mls.py
ipython mlr.py
spark-submit --master local[*] fw.py 
spark-submit --master local[2] totaltest1.py 
spark-submit --master local[2] totaltest2.py 
