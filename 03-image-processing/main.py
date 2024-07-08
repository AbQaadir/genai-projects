from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

def load_images(image_paths):
    """Load images from given paths."""
    return [Image.open(path) for path in image_paths]

def load_object_detector(model_path):
    """Load the object detection model from the given path."""
    return pipeline("object-detection", model=model_path)

def detect_objects(detector, images):
    """Detect objects in a list of images using the given detector."""
    results = []
    for image in images:
        results.append(detector(image))
    return results

def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for detection in detections:
        box = detection['box']
        label = detection['label']
        score = detection['score']
        
        draw.rectangle(
            [(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])],
            outline="red",
            width=3
        )
        draw.text(
            (box['xmin'], box['ymin'] - 10),
            f"{label} ({score:.2f})",
            fill="red",
            font=font
        )
    
    return image

def main():
    model_path = "model/snapshots/1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
    image_paths = ["image/1.jpeg", "image/2.jpg"]
    
    images = load_images(image_paths)
    object_detector = load_object_detector(model_path)
    detection_results = detect_objects(object_detector, images)
    
    for i, (image, detections) in enumerate(zip(images, detection_results)):
        image_with_detections = draw_detections(image, detections)
        image_with_detections.show()  # Display the image with detections
        image_with_detections.save(f"output-images/output_{i}.jpg")  # Save the image with detections

if __name__ == "__main__":
    main()
