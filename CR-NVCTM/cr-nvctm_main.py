# -*- coding: utf-8 -*-

import subprocess
import os

print("--------------------------------execution NVCTM activation function ---------------------------------------------")
non_linearity = ['sigmoid','relu']
for activation in non_linearity:
	for i in range(3):
		print("--------------------------------nr_topics: 100 activation:" + activation+ " execution: "+ str(i)+" ---------------------------------------------")
		name_file = "topics_nvctm_nr_topics_100_activation_"+ activation+"_"+str(i)+".txt"
		os.system("python cr-nvctm.py --n_topic 100 --non_linearity "+activation+" --topic_file="+name_file)

print("--------------------------------execution NVCTM nr_topics ---------------------------------------------")
nr_topics_test = [10,20,30,40,50,100]
for nr_topic in nr_topics_test:
	for i in range(3):
		print("--------------------------------nr_topics: " + str(nr_topic) + " execution: " + str(i)+" ---------------------------------------------")
		name_file = "topics_nvctm_nr_topics_" + str(nr_topic) + "_" + str(i)+".txt"
		os.system("python cr-nvctm.py --n_topic=" + str(nr_topic) +" --topic_file="+name_file)


print("--------------------------------execution NVCTM learning rate ---------------------------------------------")
learning_rates = [0.001,0.01]
for lr in learning_rates:
	for i in range(3):
		print("--------------------------------nr_topics: 100 learning rate:" + str(lr)+ " execution: "+ str(i)+" ---------------------------------------------")
		name_file = "topics_nvctm_nr_topics_100_lr_"+ str(lr)+"_"+str(i)+".txt"
		os.system("python cr-nvctm.py --n_topic 100 --learning_rate "+str(lr)+" --topic_file="+name_file)



print("--------------------------------execution NVCTM housholders ---------------------------------------------")
n_householders = [50,100]
for n in n_householders:
	for i in range(3):
		print("--------------------------------nr_topics: 100 n_householders:" + str(n) + " execution: "+ str(i)+" ---------------------------------------------")
		name_file = "topics_nvctm_nr_topics_100_housholders_"+ str(n)+"_"+str(i)+".txt"
		os.system("python cr-nvctm.py --n_topic 100 --n_householder "+str(n)+" --topic_file="+name_file)