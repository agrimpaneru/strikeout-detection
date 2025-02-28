import numpy as np

def sort_boxes(boxes):
    """Sort bounding boxes in reading order"""
    boxes_array = np.array(boxes)
    y_coords = boxes_array[:, 1]
    sorted_indices = np.argsort(y_coords)
    sorted_boxes = boxes_array[sorted_indices]

    final_sorted = []
    current_row = []
    y_threshold = 20
    current_y = None

    for box in sorted_boxes:
        x1, y1, x2, y2 = box
        if current_y is None or abs(y1 - current_y) > y_threshold:
            if current_row:
                current_row.sort(key=lambda b: b[0])  
                final_sorted.extend(current_row)
            current_row = [box]
            current_y = y1
        else:
            current_row.append(box)
    
    if current_row:
        current_row.sort(key=lambda b: b[0])
        final_sorted.extend(current_row)

    return [b.tolist() for b in final_sorted]
