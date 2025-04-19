from vizor.core import Tracks
from vizor.refiners.model import PrimaryModel

class Ultralytics(PrimaryModel):
    def to_tracks(self, inp, *args, **kwargs):
        return Tracks(inp.numpy()[:, [0,1,2,3,5,6,4]])

class PicklePrimary(PrimaryModel):
    """Return output from a pickle file."""
    def __init__(self, file):
        super().__init__()
        import pickle
        with open(file, "rb") as f:
            self.tracks = pickle.load(f)

    def to_tracks(self, inp, *args, **kwargs):
        return Tracks(inp.numpy()[:, [0,1,2,3,5,6,4]])

    def __iter__(self):
        for pred in self.tracks:
            yield self.to_tracks(pred)
