"""
Event Detector node implementation.

dotX Automation s.r.l. <info@dotxautomation.com>

May 6, 2026
"""

# Copyright 2025 dotX Automation s.r.l.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from threading import Thread, Lock
from event_detector.event_detector_utils import AtomicBool

from dua_node_py.dua_node import NodeBase
import dua_qos_py.dua_qos_besteffort as dua_qos_besteffort

from event_camera_msgs.msg import EventPacket
from event_camera_py import Decoder

from collections import deque
import time
from prophesee_event_msgs import msg
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from omegaconf import OmegaConf
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class EventDetectorNode(NodeBase):
    """
    Event Detector node implementation.
    """

    def __init__(self, node_name: str, verbose: bool = False) -> None:
        """
        Constructor.

        :param node_name: Name of the node.
        :param verbose: Verbosity flag.
        """
        super().__init__(node_name, verbose)

        self.declare_parameter('weights_path', '/home/neo/workspace/logs/data/dagr_s_50.pth')
        self.declare_parameter('config_path', '/opt/dagr/config/dagr-s-dsec.yaml')
        self.declare_parameter('use_image', True)
        self.declare_parameter('img_net', 'resnet50')
        self.declare_parameter('no_eval', True)
        self.declare_parameter('batch_size', 1)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('event_window_us', 50000)
        self.declare_parameter('inference_rate_hz', 20.0)

        self.init_atomics()
        self.init_detector()

        if self._autostart:
            self.activate()

        self.get_logger().info('Node initialized')

    def __del__(self) -> None:
        """
        Destructor.
        """
        self.cleanup()

    def cleanup(self) -> None:
        """
        Cleanup.
        """
        self.deactivate()

    def init_atomics(self) -> None:
        """
        Init atomics.
        """
        self._running = AtomicBool(initial=False)
        self._event_buffer = deque(maxlen=1)
        self._worker = None
        self._buffer_lock = Lock()
        self._latest_image = None
        self._image_lock = Lock()
        self._bridge = CvBridge()

    def init_detector(self) -> None:
        self._decoder = Decoder()

        weights_path = self.get_parameter('weights_path').value
        config_path = self.get_parameter('config_path').value
        height = self.get_parameter('camera_height').value
        width = self.get_parameter('camera_width').value

        if not weights_path or not config_path:
            raise RuntimeError('weights_path and config_path must be set')

        # Carica YAML
        args = OmegaConf.load(config_path)

        # Aggiungi i campi che lo script CLI passava come flag
        overrides = OmegaConf.create({
        # store_true: tutti False tranne quelli che vuoi attivi
            'use_image': True,
            'no_events': False,
            'pretrain_cnn': False,
            'keep_temporal_ordering': False,
            'no_eval': True,
            'run_test': False,
            'check_consistency': False,
            'dense': False,
            # CLI overrides (corrispondono ai flag del run_test_interframe.py)
            'img_net': 'resnet50',
            'batch_size': 1,
            'checkpoint': str(weights_path),
        })
        args = OmegaConf.merge(args, overrides)

        # Adesso DAGR ha tutti i campi che cerca
        model = DAGR(args, height=height, width=width).cuda()
        ema = ModelEMA(model)

        checkpoint = torch.load(weights_path, weights_only=False, map_location='cuda')
        ema.ema.load_state_dict(checkpoint['ema'])
        ema.ema.cache_luts(radius=args.radius, height=height, width=width)
        ema.ema.eval()

        self._model = ema.ema
        self._height = height
        self._width = width
        self.get_logger().info(f'DAGR loaded from {weights_path}')

    def init_subscribers(self) -> None:
        """
        Init subscribers.
        """
        self._event_packet_subscriber = self.dua_create_subscription(
            EventPacket,
            '/event_packet',
            self.event_packet_callback,
            dua_qos_besteffort.get_datum_qos())

        self._image_subscriber = self.dua_create_subscription(
            Image,
            '/v4l2_camera_driver/camera/image_rect_color',
            self.image_callback,
            dua_qos_besteffort.get_datum_qos())

    def activate(self) -> None:
        """
        Function to activate the Event Detector node.
        """
        self._running.store(True)
        self._worker = Thread(target=self.worker_thread_routine)
        self._worker.start()

        self.get_logger().warn('Event Detector ACTIVATED')

    def deactivate(self) -> None:
        """
        Function to deactivate the Event Detector node.
        """
        self._running.store(False)
        if self._worker is not None:
            self._worker.join()

        self.get_logger().warn('Event Detector DEACTIVATED')

    def event_packet_callback(self, msg: EventPacket) -> None:
        """Decode e accumula. NIENTE inferenza qui."""
        self._decoder.decode(msg)
        events = self._decoder.get_cd_events()
        if len(events) == 0:
            return
        with self._buffer_lock:
            self._event_buffer.append(events)

    def image_callback(self, msg: Image) -> None:
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(
                f'CvBridge failed: {e}',
                throttle_duration_sec=2.0)
            return

        if cv_img.shape[:2] != (self._height, self._width):
            cv_img = cv2.resize(
                cv_img, (self._width, self._height),
                interpolation=cv2.INTER_LINEAR)

        img = cv_img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(img).unsqueeze(0).contiguous().cuda(non_blocking=True)

        with self._image_lock:
            self._latest_image = tensor

    def _events_to_graph(self, events) -> Data:
        cam_h = self.get_parameter('camera_height').value
        cam_w = self.get_parameter('camera_width').value
        window_us = self.get_parameter('event_window_us').value

        EVK4_H, EVK4_W = 720, 1280
        sx = cam_w / EVK4_W
        sy = cam_h / EVK4_H

        x = events['x'].astype(np.float32) * sx
        y = events['y'].astype(np.float32) * sy
        t = events['t'].astype(np.float32)
        p = events['p'].astype(np.float32)

        t_norm = (t - t.min()) / max(t.max() - t.min(), 1.0)
        x_norm = x / cam_w
        y_norm = y / cam_h

        pos = torch.from_numpy(np.stack([x_norm, y_norm, t_norm], axis=1)).float().cuda()
        feat = torch.from_numpy(2.0 * p - 1.0).float().unsqueeze(1).cuda()

        data = Data(x=feat, pos=pos)
        data.height = cam_h
        data.width = cam_w
        data.time_window = window_us

        with self._image_lock:
            if self._latest_image is not None:
                data.image = self._latest_image
            else:
                self.get_logger().warn(
                    'No image received yet, falling back to zeros',
                    throttle_duration_sec=2.0)
                data.image = torch.zeros(1, 3, self._height, self._width, device='cuda')

        return Batch.from_data_list([data])

    def worker_thread_routine(self) -> None:
        """Inferenza a frequenza fissa."""
        rate_hz = self.get_parameter('inference_rate_hz').value
        period = 1.0 / rate_hz
        window_us = self.get_parameter('event_window_us').value
        min_events = 500

        self.get_logger().info(f'Worker started @ {rate_hz} Hz')

        while self._running.load():
            loop_start = time.monotonic()

            with self._buffer_lock:
                if not self._event_buffer:
                    packets = []
                else:
                    packets = list(self._event_buffer)
                    self._event_buffer.clear()

            if not packets:
                self.get_logger().info('Event buffer empty', throttle_duration_sec=2.0)
                time.sleep(period)
                continue

            events = np.concatenate(packets)
            if len(events) < min_events:
                self.get_logger().info(
                    f'Skipping: only {len(events)} events',
                    throttle_duration_sec=2.0)
                time.sleep(period)
                continue

            t_max = events['t'].max()
            cutoff = t_max - window_us
            events = events[events['t'] >= cutoff]
            if len(events) < min_events:
                time.sleep(period)
                continue

            try:
                graph = self._events_to_graph(events)
                t0 = time.monotonic()
                with torch.no_grad():
                    output = self._model(graph)
                torch.cuda.synchronize()
                t_inference = (time.monotonic() - t0) * 1000

                # Output noto: list[list[dict('boxes','scores','labels')]]
                num_boxes = 0
                max_score = 0.0
                try:
                    det = output[0][0]
                    num_boxes = int(det['boxes'].shape[0])
                    if num_boxes > 0:
                        max_score = float(det['scores'].max().item())
                except (IndexError, KeyError, TypeError):
                    pass

                self.get_logger().info(
                    f'Inference: {len(events)} ev, {t_inference:.1f} ms, '
                    f'{num_boxes} boxes, max_score={max_score:.3f}',
                    throttle_duration_sec=1.0)

            except Exception as e:
                import traceback
                self.get_logger().error(
                    f'Inference failed: {e}\n{traceback.format_exc()}',
                    throttle_duration_sec=5.0)

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period - elapsed))

        self.get_logger().info('Worker stopped')
