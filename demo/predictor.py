"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/predictor.py
"""
import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch
import itertools
import numpy as np
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor as d2_defaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
import detectron2.utils.visualizer as d2_visualizer
import pdb

class DefaultPredictor(d2_defaultPredictor):

    def set_metadata(self, metadata):
        self.model.set_metadata(metadata)


class OpenVocabVisualizer(Visualizer):
    def draw_panoptic_seg(self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7):
        """
        Draw panoptic prediction annotations or results.

        Args:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
                segment.
            segments_info (list[dict] or None): Describe each segment in `panoptic_seg`.
                If it is a ``list[dict]``, each dict contains keys "id", "category_id".
                If None, category id of each pixel is computed by
                ``pixel // metadata.label_divisor``.
            area_threshold (int): stuff segments with less than `area_threshold` are not drawn.

        Returns:
            output (VisImage): image object with visualizations.
        """
        pred = d2_visualizer._PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))
        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            text = self.metadata.stuff_classes[category_idx].split(',')[0]
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=d2_visualizer._OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        # draw mask for all instances second
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]

        try:
            scores = [x["score"] for x in sinfo]
        except KeyError:
            scores = None
        stuff_classes = self.metadata.stuff_classes
        stuff_classes = [x.split(',')[0] for x in stuff_classes]
        labels = d2_visualizer._create_text_labels(
            category_ids, scores, stuff_classes, [x.get("iscrowd", 0) for x in sinfo]
        )

        try:
            colors = [
                self._jitter([x / 255 for x in self.metadata.stuff_colors[c]]) for c in category_ids
            ]
        except AttributeError:
            colors = None
        self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

        return self.output


    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.6):
        stuff_classes = self.metadata.stuff_classes
        stuff_classes = [i.split(',')[0] for i in stuff_classes]
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        for label in filter(lambda l: l < len(stuff_classes), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = stuff_classes[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=(1.0, 1.0, 240.0 / 255),
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output
    

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """

        coco_metadata = MetadataCatalog.get("openvocab_coco_2017_val_panoptic_with_sem_seg")
        ade20k_metadata = MetadataCatalog.get("openvocab_ade20k_panoptic_val")
        # lvis_classes = open("./maft/data/datasets/lvis_1203_with_prompt_eng.txt", 'r').read().splitlines()
        # lvis_classes = [x[x.find(':')+1:] for x in lvis_classes]
        # lvis_colors = list(
        #     itertools.islice(itertools.cycle(coco_metadata.stuff_colors), len(lvis_classes))
        # )
        # # rerrange to thing_classes, stuff_classes
        # coco_thing_classes = coco_metadata.thing_classes
        # coco_stuff_classes = [x for x in coco_metadata.stuff_classes if x not in coco_thing_classes]
        # coco_thing_colors = coco_metadata.thing_colors
        # coco_stuff_colors = [x for x in coco_metadata.stuff_colors if x not in coco_thing_colors]
        # ade20k_thing_classes = ade20k_metadata.thing_classes
        # ade20k_stuff_classes = [x for x in ade20k_metadata.stuff_classes if x not in ade20k_thing_classes]
        # ade20k_thing_colors = ade20k_metadata.thing_colors
        # ade20k_stuff_colors = [x for x in ade20k_metadata.stuff_colors if x not in ade20k_thing_colors]

        # user_classes = []
        # user_colors = [random_color(rgb=True, maximum=1) for _ in range(len(user_classes))]

        # stuff_classes = coco_stuff_classes + ade20k_stuff_classes
        # stuff_colors = coco_stuff_colors + ade20k_stuff_colors
        # thing_classes = user_classes + coco_thing_classes + ade20k_thing_classes + lvis_classes
        # thing_colors = user_colors + coco_thing_colors + ade20k_thing_colors + lvis_colors

        # test
        # user_classes = open("./maft/data/datasets/lvis_1203_with_prompt_eng.txt", 'r').read().splitlines()
        # user_classes = [x[x.find(':')+1:] for x in user_classes]
        # user_classes = ["keyboard and mouse"]
        # user_classes = ["dog and person"]

        # thing_dataset_id_to_contiguous_id = {x: x for x in range(len(user_classes))}
        # DatasetCatalog.register(
        #     "openvocab_dataset", lambda x: []
        # )
        # self.metadata = MetadataCatalog.get("openvocab_dataset").set(
        #     stuff_classes=user_classes,
        #     stuff_colors=[np.random.randint(256, size=3).tolist() for _ in range(len(user_classes))],
        #     thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        # )
        #print("self.metadata:", self.metadata)
        
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.parallel = parallel

        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        self.metadata = None  # will be assigned per image
        # self.predictor.set_metadata(...) is done dynamically in run_on_image()

    def run_on_image(self, image, user_classes):
        """
        Args:
            image (np.ndarray): BGR image
            user_classes (List[str]): e.g., ["dog and person"]

        Returns:
            predictions, vis_output, pooled_score_map
        """
        from detectron2.data import MetadataCatalog, DatasetCatalog

        # Create a unique dataset name using command string hash
        dataset_name = f"openvocab_dynamic_{abs(hash(user_classes[0])) % 10**8}"

        if dataset_name not in MetadataCatalog.list():
            DatasetCatalog.register(dataset_name, lambda: [])
            MetadataCatalog.get(dataset_name).set(
                stuff_classes=user_classes,
                stuff_colors=[[255, 0, 0]],
                thing_dataset_id_to_contiguous_id={0: 0}
            )

        self.metadata = MetadataCatalog.get(dataset_name)
        self.predictor.set_metadata(self.metadata)

        predictions = self.predictor(image)
        image_rgb = image[:, :, ::-1]
        visualizer = OpenVocabVisualizer(image_rgb, self.metadata, instance_mode=self.instance_mode)

        pooled_score_map = None

        if "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"].to(self.cpu_device)  # shape: [C, H, W]
            score_map = sem_seg[0]  # Only one class, take channel 0

            # 2. Apply 8×8 adaptive average pooling
            import torch.nn.functional as F
            pooled = F.adaptive_avg_pool2d(score_map.unsqueeze(0).unsqueeze(0), (8, 8))
            pooled_score_map = pooled.squeeze().cpu().numpy()

            # 3. Threshold and create binary mask for visualization
            confidence_threshold = 0.6
            mask = score_map > confidence_threshold
            height, width = score_map.shape
            segmentation = torch.full((height, width), 255, dtype=torch.uint8)
            segmentation[mask] = 0  # 0 is the index of your only class
            vis_output = visualizer.draw_sem_seg(segmentation)

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output, pooled_score_map

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
