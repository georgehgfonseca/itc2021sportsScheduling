powershell "python main.py ../../data/ITC2021_Middle_9.xml -initsoln_file ../../data/ITC2021_Middle_9Soln.xml -output ../../data/ITC2021_Middle_9Soln.xml -emphasis 1 -timelimit 7200000 -fixopt_timelimit 200 -fixopt_n 10 -fixopt_neighborhood s -local_branching 0 -threads 1 -verbose 0 | tee ../../data/ITC2021_Middle_9Log.txt"