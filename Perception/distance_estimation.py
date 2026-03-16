import cv2
from ultralytics import YOLO

def estimate_cone_distances(
    image_path: str,
    model_path: str = "best.pt",
    cone_height_cm: float = 30.0,
    focal_length_mm: float = 1000.0,
    save_output: bool = True
):
    """
    Detect cones in an image and estimate their distance using YOLO and pinhole camera geometry.

    Parameters:
    - image_path: path to the input image
    - model_path: path to YOLO weights
    - cone_height_cm: real-world height of a cone (cm)
    - focal_length_mm: camera focal length (mm)
    - save_output: if True, save annotated output image as 'output.jpg'

    Returns:
    - distances_dict: dictionary of cones with their distances in meters
    """

    model = YOLO(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(img)

    cones = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        class_ids = result.boxes.cls.cpu().numpy()  
        names = model.names  # get class names --> colour names

        for box, cls_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            h_pixels = y2 - y1
            if h_pixels == 0:
                continue  

            # Depth calculation using pinhole camera geometry
            # d = (H * f) / h
            # H in cm, f in mm, h in pixels → convert distance --> meters
            distance_m = (cone_height_cm * focal_length_mm) / h_pixels / 100.0

            cones.append({
                "color": names[int(cls_id)],
                "distance": distance_m,
                "x": x1,
                "box": (x1, y1, x2, y2)
            })

    # Sort cones from left to right
    cones.sort(key=lambda c: c["x"])

    # labels per color (eg. blue1)
    color_counts = {}
    distances_dict = {}

    for c in cones:
        color = c["color"]
        color_counts[color] = color_counts.get(color, 0) + 1
        label = f"{color}{color_counts[color]}"
        distances_dict[label] = c["distance"]


        x1, y1, x2, y2 = c["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)

        label_text = f"{c['distance']:.2f} m"

        font_scale = 0.4
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


    # Print distances
    print("Detected cone distances (left --> right):")
    for label, dist in distances_dict.items():
        print(f"{label}: {dist:.2f} m")

    # Show the annotated image
    cv2.imshow("Detected Cones", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output
    if save_output:
        cv2.imwrite("output.jpg", img)
        print("Annotated image saved as 'output.jpg'")

    return distances_dict

if __name__ == "__main__":
    estimate_cone_distances("image.png", model_path="best.pt")