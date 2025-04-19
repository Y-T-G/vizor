from vizor.core import Preds, Outs
from vizor.refiners.model import SecondaryModel
from .model import TransformersModel
import numpy as np


class Ultralytics(SecondaryModel):
    def to_preds(preds):
        return Preds(preds[0].boxes.data.numpy())

class Florence(TransformersModel):
    def __init__(self, model, classes, task="causallm", parser=None, task_prompt="<CAPTION_TO_PHRASE_GROUNDING>"):
        super().__init__(model, task, parser, task_prompt)
        self.classes = classes
        self.classes2id = {v:k for k,v in classes.items()}

    def to_preds(self, preds):
        """Florence output to Preds."""
        out = list(preds.values())[0]
        bboxes = out["bboxes"]
        classes = [[self.names2id[label]] for label in out["labels"]]
        scores = [[1.0]] * len(classes)
        out = np.hstack((bboxes, scores, classes), dtype=np.float32)
        preds = Preds(out)
        return preds


class PickleSecondary(SecondaryModel):
    """Return output from a pickle file."""
    def __init__(self, file):
        super().__init__()
        import pickle
        with open(file, "rb") as f:
            self.preds = pickle.load(f)

    def to_preds(self, inp, *args, **kwargs):
        # return Preds(inp[:, [0,1,2,3,5,6,4]])
        return Preds(inp)

    def __iter__(self):
        for pred in self.preds:
            yield self.to_preds(pred)