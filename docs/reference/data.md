# Data

Everything moves through one layout, a float32 `(N, 7)` array of
`[x1, y1, x2, y2, conf, cls, id]`. `Tracks` wraps it, `Track` is one row of it,
and `Preds` is the id-less version used for secondary output.

::: vizor.boxes.Tracks

::: vizor.boxes.Track

::: vizor.boxes.Preds

::: vizor.boxes.iou
