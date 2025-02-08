def crop_box(img, box, border=0.1):
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    margin = int(border * min(box_width, box_height))
    x1m, y1m = max(0, x1 - margin), max(0, y1 - margin)
    x2m, y2m = min(img.shape[1], x2 + margin), min(img.shape[0], y2 + margin)
    return img[y1m:y2m, x1m:x2m]