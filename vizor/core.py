from powerboxes import iou_distance
from dataclasses import dataclass
import numpy as np

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
        data in [x1,y1,x2,y2,score,label, tid] format
        """
        self.data = tracks
        self.tracks = [Track(track[:4], *track[4:7]) for track in tracks]

    def __getitem__(self, idx):
        return self.tracks[idx]

    def __setitem__(self, idx, track):
        self.tracks[idx] = track

    def __iter__(self):
        return iter(self.tracks)

    def __repr__(self):
        return str(self.tracks)


class Preds(Tracks):
    def __init__(self, preds):
        """
        data in [x1,y1,x2,y2,score,label] format
        """
        self.data = preds
        self.tracks = [Track(track[:4], *track[4:6], -1) for track in preds]
    

class Refiner:
    # This is the main class to be used for running oomph. Users are expected
    # to provide the primary and secondary models with the required methods and
    # also a tracker with the required method.
    def __init__(self, model, in_transform, out_transform, conf=0.9, mode="instance"):
        # The model is only used when the conditions for the tracks are met.
        # For example, new tracks should run secondary model for verification
        # of the prediction.
        self.model = model
        # "image" mode makes an inference on the whole image and "instance"
        # mode only makes an inference on the individual instances.
        self.mode = mode
        # Keeps track of refined tracks
        self.refined = {}
        # minimum confidence to trigger secondary inference
        self.min_conf = conf
        # minimum iou used to match secondary detections with primary detections
        self.min_iou = 0.5
        # Converstion funcs
        self.in_transform = in_transform
        self.out_transform = out_transform

    def run(self, img, tracks):
        # use the whole image for secondary inference
        if self.mode == "image":
            # If any track has less that this conf, they will be processed by the secondary model
            if any([track.conf <= self.min_conf for track in tracks]):
                # preds would contain the results from secondary inference
                preds = Preds(self.model.predict(img)[0].boxes.data.numpy())
                # we need to match the secondary results with primary results
                iou_tracks = 1 - iou_distance(tracks.data[:, :4], preds.data[:, :4])
                for i, iou_track in enumerate(iou_tracks):
                    j = iou_track.argmax() # The IOU of the best matched box
                    track = tracks[i]
                    if iou_track[j] >= self.min_iou:
                        # Update with rectified predictions
                        pred = preds[j]
                        track.box = pred.box
                        # Store the class so that we can map it even without secondary model run
                        self.refined[track.id] = cls = pred.cls
                    # Updates with stored classes; could also be from previous frames
                    track.cls = self.refined.get(track.id, track.cls)
                    tracks[i] = track
                    
        # use crops for secondary inference
        elif self.mode == "instance":
            for (x1, y1, x2, y2) in tracks.xyxy:
                # If any track has less that this conf, they will be processed by the secondary model
                if track.conf > self.min_conf:
                    continue
                crop = img[y1:y2, x1:x2]
                # pred would contain the result from secondary inference
                pred = self.model.predict(crop)
                # Update with rectified predictions
                track.update(pred)
                # Store the class so that we can map it even without secondary model run
                self.refined[track.id] = pred.cls
                if track.id in self.refined:
                    cls = self.refined[track.id]
                track.cls = cls
        return tracks
