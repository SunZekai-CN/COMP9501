### Launch the Stage simulator
Launch the Stage simulator (We assume that you already have installed the Stage simulator. ) by: 
```
   ros2 run stage_ros stageros -u src/stage_ros/world/task_5.world
```

* 'src/stage\_ros/world/task\_5.world' 
is the path of world file, which describes the training environment consisting of a map, robots, obstacles and so on. If you want the robots move in a new environment, you should make a new world file to describe the environment. 
* We use this world file just because the environment in this word file satisfy the requirement of our task on the number of robots,obstacles and so on.
### Wake up each robot
Follow the guide of 'config.yaml.sample' and fill the file 'config-deploy.yaml' with the IP addresses of each robot process. Then you can start the leader process by :
```
    python3 oracle.py config-deploy.yaml
```
Then start each robot process on each device by :
```
    python3 worker.py config-deploy.yaml {each_id}
```
* each\_id is The index number of each robot. in this task, we have 5 robots, so each\_id should be 0 to 4. 
### Training
Start parameter server by:
```
    python3 leader.py config-deploy.yaml 
```
and start each training worker by:
```
    python3 train_worker.py {ps_ip} {ps_port}
```
* ps\_ip and ps\_port are the ip address of parameter server. 
### Congratulations
Then, you will see the robots in the Stage simulator move and the training start.