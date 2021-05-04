import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
import array
import utils
import sys
import socket
import pickle
import time
import threading
import queue
import logging
logger = logging.getLogger('WorkerCycle')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s: %(message)s'))
logger.addHandler(ch)

F=30
class Sensor:
    """Get perception data from stream continously, but only keep the latest one."""

    def __init__(self, mstream):
        self.mstream = mstream

        self.data = None
        self.queue = queue.Queue()

        def update_data_task():
            while True:
                data = pickle.loads(mstream.recv())
                self.queue.put(data)
        self.update_data_thread = threading.Thread(target=update_data_task)
        self.update_data_thread.start()

    def read(self):
        return self.queue.get()


class Actuator:
    def __init__(self, mstream):
        self.mstream = mstream

    def write(self, data):
        self.mstream.send(pickle.dumps(data))


class PerceptionNode(Node):
    def __init__(self, my_id, sensor: Sensor):
        super().__init__(f'perception_node_{my_id}')
        self.my_id = my_id

        self.publisher = self.create_publisher(
            UInt8MultiArray, f'{"urg_" if self.my_id != 0 else ""}perception_{my_id}', utils.get_ros_qos_profile())

        def perception_task():
            prev_time = None
            max_diff = 0
            while True:
                data = sensor.read()
                if data is None:
                    continue
                # time.sleep(0.0035*my_id)
                msg = UInt8MultiArray()
                msg.data = array.array('B', pickle.dumps((data, time.time(), bytes(9*1024))))
                self.publisher.publish(msg)
                # check if periodic
                # assert prev_time is None or cur_time - prev_time < 200e-3
                cur_time = time.time()
                if prev_time is not None:
                    max_diff = max(max_diff, abs(cur_time - prev_time - 1/F))
                    logger.debug(
                        f'perc cur={cur_time - prev_time}, max_diff={max_diff}')
                prev_time = cur_time
        self.thread = threading.Thread(target=perception_task)

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()


class ActionNode(Node):
    def __init__(self, my_id, actuator: Actuator):
        super().__init__(f'action_node_{my_id}')
        self.my_id = my_id

        def handler(msg):
            ts_receive = time.time()
            data, ts, pts = pickle.loads(msg.data)
            actuator.write(data)

            #logger.info(
            #    f'action delay of {my_id} {(ts_receive - ts)*1e3:.3f}ms')
            #logger.info(f'perception-control delay of {my_id} {(ts_receive - pts)*1e3:.3f}ms')
        self.subscriber = self.create_subscription(
            UInt8MultiArray,
            # f'{"urg_" if self.my_id != 0 else ""}action_{my_id}',
            f'action_{my_id}',
            handler, utils.get_ros_qos_profile())

        def spin_task():
            rclpy.spin(self)
        self.thread = threading.Thread(target=spin_task)

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()


if __name__ == '__main__':
    config_path = sys.argv[1]
    my_id = int(sys.argv[2])
    with open(config_path) as conf:
        config = yaml.load(conf, yaml.FullLoader)
    oracle_ip = config['sim_ips'][0]
    oracle_port = config['oracle_port']

    rclpy.init(args=['--ros-args', '--log-level', 'info'])

    # connect to leader robot
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((oracle_ip, oracle_port))
            leader_msg_stream = utils.MessageStream(sock)
            break
        except ConnectionRefusedError:
            time.sleep(0.5)

    sensor = Sensor(leader_msg_stream)
    perception_node = PerceptionNode(my_id, sensor)
    perception_node.start()

    actuator = Actuator(leader_msg_stream)
    action_node = ActionNode(my_id, actuator)
    action_node.start()

    perception_node.join()
    action_node.join()
