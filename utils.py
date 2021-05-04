import socket
import pickle
import threading
import time
import math
import queue
import ctypes

from rclpy.qos import QoSProfile, QoSReliabilityPolicy

import logging
logger = logging.getLogger('Utils')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s: %(message)s'))

MAX_RECV_SIZE = 4*1024

def get_ros_qos_profile():
    qos_profile = QoSProfile(depth=10)
    qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
    return qos_profile


def calibrate_freq(freq, base_time):
    cur_time = time.monotonic()
    sleep_time = math.ceil((cur_time-base_time)/(1/freq))*(1/freq) + base_time - cur_time
    time.sleep(sleep_time)


class TCPInfo(ctypes.Structure):
    """TCP_INFO struct in linux 5.4.84
    see /usr/include/linux/tcp.h for details.

    This snippet is adapted from https://github.com/ModioAB/snippets/blob/master/src/tcp_info.py
    """

    __u8 = ctypes.c_uint8
    __u32 = ctypes.c_uint32
    __u64 = ctypes.c_uint64

    _fields_ = [
        ("tcpi_state", __u8),
        ("tcpi_ca_state", __u8),
        ("tcpi_retransmits", __u8),
        ("tcpi_probes", __u8),
        ("tcpi_backoff", __u8),
        ("tcpi_options", __u8),
        ("tcpi_snd_wscale", __u8, 4), ("tcpi_rcv_wscale", __u8, 4),
        ("tcpi_delivery_rate_app_limited", __u8, 1),

        ("tcpi_rto", __u32),
        ("tcpi_ato", __u32),
        ("tcpi_snd_mss", __u32),
        ("tcpi_rcv_mss", __u32),

        ("tcpi_unacked", __u32),
        ("tcpi_sacked", __u32),
        ("tcpi_lost", __u32),
        ("tcpi_retrans", __u32),
        ("tcpi_fackets", __u32),

        # Times
        ("tcpi_last_data_sent", __u32),
        ("tcpi_last_ack_sent", __u32),
        ("tcpi_last_data_recv", __u32),
        ("tcpi_last_ack_recv", __u32),

        # Metrics
        ("tcpi_pmtu", __u32),
        ("tcpi_rcv_ssthresh", __u32),
        ("tcpi_rtt", __u32),
        ("tcpi_rttvar", __u32),
        ("tcpi_snd_ssthresh", __u32),
        ("tcpi_snd_cwnd", __u32),
        ("tcpi_advmss", __u32),
        ("tcpi_reordering", __u32),

        ("tcpi_rcv_rtt", __u32),
        ("tcpi_rcv_space", __u32),

        ("tcpi_total_retrans", __u32),

        ("tcpi_pacing_rate", __u64),
        ("tcpi_max_pacing_rate", __u64),
        # RFC4898 tcpEStatsAppHCThruOctetsAcked
        ("tcpi_bytes_acked", __u64),
        # RFC4898 tcpEStatsAppHCThruOctetsReceived
        ("tcpi_bytes_received", __u64),
        # RFC4898 tcpEStatsPerfSegsOut
        ("tcpi_segs_out", __u32),
        # RFC4898 tcpEStatsPerfSegsIn
        ("tcpi_segs_in", __u32),

        ("tcpi_notsent_bytes", __u32),
        ("tcpi_min_rtt", __u32),
        ("tcpi_data_segs_in", __u32),
        ("tcpi_data_segs_out", __u32),

        ("tcpi_delivery_rate", __u64),

        ("tcpi_busy_time", __u64),
        ("tcpi_rwnd_limited", __u64),
        ("tcpi_sndbuf_limited", __u64),

        ("tcpi_delivered", __u32),
        ("tcpi_delivered_ce", __u32),

        ("tcpi_bytes_sent", __u64),
        ("tcpi_bytes_retrans", __u64),
        ("tcpi_dsack_dups", __u32),
        ("tcpi_reord_seen", __u32),

        ("tcpi_rcv_ooopack", __u32),

        ("tcpi_snd_wnd", __u32),
    ]
    del __u8, __u32, __u64


class MutexSocket:
    TIME_SLICE = 5
    def __init__(self, sock: socket.socket):
        # connect to lock server
        lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        lock_sock.connect(('10.42.0.1', 9999))
        lock_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lock_stream = MessageStream(lock_sock)

        self.sock = sock

        self.send_queue = queue.Queue()
        def control_task():
            done = False
            while not done:
                # wait for permit
                self.lock_stream.send(pickle.dumps(('acquire', None)))
                resp, token = pickle.loads(self.lock_stream.recv())
                assert resp == 'permit'
                # resume
                time_end = time.monotonic() + self.TIME_SLICE
                self.resume()
                # keep checking
                while time.monotonic() < time_end:
                    time.sleep(0.100)  # 100ms
                    # if done, release and return
                    if self.empty():
                        done = True
                        self.lock_stream.send(pickle.dumps(('release', token)))
                        break
                self.pause()
        def send_task():
            while True:
                data = self.send_queue.get()
                # pause socket
                self.pause()
                # write socket, use another thread, because a large amount data would block
                tmp_thread = threading.Thread(target=lambda:self.sock.sendall(data))
                tmp_thread.start()
                # create and join the control thread
                control_thread = threading.Thread(target=control_task)
                control_thread.start()
                control_thread.join()
        self.send_thread = threading.Thread(target=send_task)
        self.send_thread.start()

    def send(self, data: bytes, *_):
        self.send_queue.put(bytearray(data))

    def recv(self, *_):
        return self.sock.recv(MAX_RECV_SIZE)

    def pause(self):
        self.sock.setsockopt(socket.IPPROTO_TCP, 38, 1)

    def resume(self):
        self.sock.setsockopt(socket.IPPROTO_TCP, 38, 0)

    def empty(self):
        buf = self.sock.getsockopt(socket.SOL_TCP, socket.TCP_INFO, ctypes.sizeof(TCPInfo))
        tcp_info = TCPInfo.from_buffer_copy(buf)
        return tcp_info.tcpi_notsent_bytes == 0


class MessageStream:
    """ Thread safe, nonblocking send, blocking recv
    """
    NUM_SIZE = 4

    def __init__(self, sock: socket.socket, mutex=False):
        self.sock = sock if not mutex else MutexSocket(sock)

        self.send_queue = queue.Queue()
        def send_task():
            while True:
                msg = self.send_queue.get()
                msize = len(msg)
                self.sock.send(msize.to_bytes(self.NUM_SIZE, 'big')+msg)
        self.send_thread = threading.Thread(target=send_task)
        self.send_thread.start()

        self.recv_queue = queue.Queue()
        def recv_task():
            buffer = bytearray()
            while True:
                # get message size
                while len(buffer) < self.NUM_SIZE:
                    buffer += self.sock.recv(MAX_RECV_SIZE)
                msize = int.from_bytes(buffer[:self.NUM_SIZE], 'big')
                buffer = buffer[self.NUM_SIZE:]
                # get message
                while len(buffer) < msize:
                    buffer += self.sock.recv(MAX_RECV_SIZE)
                msg = buffer[:msize]
                buffer = buffer[msize:]

                self.recv_queue.put(msg)
        self.recv_thread = threading.Thread(target=recv_task)
        self.recv_thread.start()

    def send(self, msg):
        self.send_queue.put(msg)
    
    def recv(self):
        return self.recv_queue.get()
