from powerboxes import iou_distance
from dataclasses import dataclass
from lru import LRU
import numpy as np

from .utils.image import crop_box, COLORS
import cv2


class BaseModel:
    # Should be marked as abstract class
    def __init__(self, model):
        self.model = model

    def predict(self, img, tracks):
        # This is a required method that the users are expected to implement for their models.
        # The input are the tracks obtained from the tracker [x1,y1,x2,y2,conf,class_id,track_id] while the output is the predictions in [x1,y1,x2,y2,conf,class_id,track_id] format.
        raise NotImplementedError

@dataclass
class Track:
    """Tracklet in Vizor refiner format."""
    box: np.ndarray
    conf: float
    cls: int
    id: int


class Tracks:
    def __init__(self, tracks):
        """
        data in [x1,y1,x2,y2,score,label,tid] format
        """
        self.data = tracks

    def __getitem__(self, idx):
        return Track(self.data[idx][:4], *self.data[idx][4:7])

    def __setitem__(self, idx, track):
        if isinstance(track, Track):
            self.data[idx] = [*track.box, track.conf, track.cls, track.id]
        else:
            self.data[idx] = track

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return str(list(self))


class Preds(Tracks):
    def __init__(self, preds):
        """
        data in [x1,y1,x2,y2,score,label] format
        """
        # Add -1 as track id for each prediction
        super().__init__(preds if preds.shape[-1] == 7 else np.column_stack([preds, np.full(len(preds), -1, dtype=preds.dtype)]))

class Outs(Tracks):
    def __init__(self, outs, classes):
        """
        refiner output in [x1,y1,x2,y2,score,label,tid] format
        """
        self.data = outs
        self.classes = classes
    
    def draw(self, img):
        for det in self.data:
            x1, y1, x2, y2, score, label, track_id = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            color = COLORS[int(label) % len(COLORS)]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # If self.classes is a list containing label names, use it; otherwise just use the label id.
            label_name = self.classes[int(label)] if isinstance(self.classes, list) and int(label) < len(self.classes) else str(int(label))
            text = f"{int(track_id)}:{label_name}:{score:.1f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)

            cv2.putText(img, text, (x1, y1 - baseline), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
        return img


class Refiner:
    # This is the main class to be used for running oomph. Users are expected
    # to provide the primary and secondary models with the required methods and
    # also a tracker with the required method.
    def __init__(self, model, conf=0.9, mode="instance", classes=None):
        # The model is only used when the conditions for the tracks are met.
        # For example, new tracks should run secondary model for verification
        # of the prediction.
        self.model = model
        # "image" mode makes an inference on the whole image and "instance"
        # mode only makes an inference on the individual instances.
        self.mode = mode
        # Keeps track of refined tracks
        self.refined = LRU(size=100)
        # minimum confidence to trigger secondary inference
        self.min_conf = conf
        # minimum iou used to match secondary detections with primary detections
        self.min_iou = 0.5
        self.classes = classes
        # Converstion funcs
        # self.in_transform = in_transform
        # self.out_transform = out_transform
        # self.to_preds = to_preds  # transform secondary model's output to Preds

        # Instance mode configs
        self.margin = 10

    def out_transform(self, out, img=None, *args, **kwargs):
        # transform the output of refiner to Outs
        return Outs(out.data, self.classes)

    def run(self, tracks, *args, img=None, preds=None, **kwargs):
        # use the whole image for secondary inference
        # tracks = self.in_transform(tracks, *args, **kwargs)
        if self.mode == "image":
            # If any track has less that this conf, they will be processed by the secondary model
            if any([track.conf <= self.min_conf for track in tracks]):
                # preds would contain the results from secondary inference
                if preds is None:
                    preds = self.model.predict(img, **kwargs)
                    # preds = self.to_preds(preds)
                # we need to match the secondary results with primary results
                if tracks.data.shape[0] and preds.data.shape[0]:
                    iou_tracks = 1 - iou_distance(tracks.data[:, :4], preds.data[:, :4])
                else:
                    iou_tracks = []
                for i, iou_track in enumerate(iou_tracks):
                    j = iou_track.argmax() # The IOU of the best matched box
                    track = tracks[i]
                    if iou_track[j] >= self.min_iou:
                        # Update with rectified predictions
                        pred = preds[j]
                        track.box = pred.box
                        track.conf = pred.conf
                        # Store the class so that we can map it even without secondary model run
                        self.refined[track.id] = cls = pred.cls
                    # Updates with stored classes; could also be from previous frames
                    track.cls = self.refined.get(track.id, track.cls)
                    tracks[i] = track
            else:
                # In case there are no low conf preds, but still apply refiner.
                for i, track in enumerate(tracks):
                    # Updates with stored classes; could also be from previous frames
                    track.cls = self.refined.get(track.id, track.cls)
                    tracks[i] = track
                    
        # use crops for secondary inference
        elif self.mode == "instance":
            for i, track in enumerate(tracks):
                if track.conf <= self.min_conf and track.id not in self.refined:
                    crop = crop_box(img, track.box.round().astype(int))
                    # pred would contain the class ID result from secondary inference
                    pred = int(self.model.predict(crop[...,::-1], **kwargs))
                    # Update with rectified predictions
                    # Store the class so that we can map it even without secondary model run
                    cls = self.refined[track.id] = pred
                elif track.id in self.refined:
                    # Get stored class ID for the same box if available
                    cls = self.refined.get(track.id, track.cls)
                track.cls = cls
                tracks[i] = track

        return self.out_transform(tracks, img, *args, **kwargs)
