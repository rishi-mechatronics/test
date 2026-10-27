import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Define COCO indices considered as "waste/recyclable items"
# 39: bottle, 41: cup, 46: banana, 47: apple, 73: book, 79: toothbrush
WASTE_CLASS_IDS = [39,41,46,47,73,79]

# this would be the only 6 items the ai can detect.
BIN_RULES = {
    "bottle": "Plastic Bin",
    "cup": "Paper Bin",
    "banana": "Compost Bin",
    "apple": "Compost Bin",
    "book": "Paper Bin",
    "toothbrush": "Plastic Bin",
}

# id item detected is less than this threshold, it would be put in the rejected bin
DEFAULT_THRESHOLD = 0.50

#defining the rejected bin
REJECT_BIN = "Reject Bin"

class WasteItem:
    '''defining the items that the ai will detect so there is the category the item will be in
    the confidence level of the ai
    and the location of the item, so it's coordinates'''
    def __init__(self, category: str, score: float, location: tuple):
        self.category = category      # The name of the class (e.g., 'bottle')
        self.score = score            # Confidence score between 0.0 and 1.0
        self.location = location      # Bounding box coordinates (x1, y1, x2, y2)

class Report:
    '''
    1. Creates an empty list to store all the sorting results the ai detects
    2. it stores the result that the ai detects in the list (Adds any new items detected)
    3. displays all the results during the detection session
    4. show summary of the detected items.'''
    def __init__(self):
        # List to store all sorting results
        self.history = []

    def add_result(self, item, category, confidence, bin_name):
        # Store the result of one sorting operation
        result = {
            "item": item,
            "category": category,
            "confidence": confidence,
            "bin": bin_name
        }
        self.history.append(result)

    def show_history(self):
        # Display all recorded results
        print("\n--- Sorting History ---")
        for result in self.history:print(f"Item: {result['category']} | Confidence: {result['confidence']:.2f} | Assigned: {result['bin']}")

    def show_summary(self):
        # Count the number of items in each bin
        bin_counts = {}
        for result in self.history:
            bin_name = result["bin"]
            bin_counts[bin_name] = bin_counts.get(bin_name, 0) + 1


        print("\n--- Sorting Summary ---")
        for bin_name, count in bin_counts.items():
            print(bin_name, ":", count, "item(s)")


class Sorter:
    '''
    1. counter for every item added to bin even rejected bin.
    2. rules for items detected
    3. Updates counter and displays message of what has been added
    4. prints out counter
    5. shows a summery of what has been added to list'''

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.counts = {bin_name: 0 for bin_name in set(BIN_RULES.values())}
        self.counts[REJECT_BIN] = 0

    def decide_bin(self, item: WasteItem):
        """Return (bin_name, reason) without changing any counters."""
        if item.category not in BIN_RULES:
            return REJECT_BIN, "No items from list have been detected."
        if item.score < self.threshold:
            return REJECT_BIN, "YOLO hasn't confidently detected item."
        return BIN_RULES[item.category], "Item has been added to bin list."

    def sort(self, item: WasteItem):
        """Sort one item: update the counter, print the message, return (bin, reason)."""
        bin_name, reason = self.decide_bin(item)
        self.counts[bin_name] += 1
        print(f"This waste is a {item.category} and will be sorted in bin {bin_name}")
        return bin_name, reason

    def show_counts(self):
        print("\n--- Bin Counters ---")
        for bin_name in sorted(self.counts):
            print(f"{bin_name}: {self.counts[bin_name]}")

    def show_summary(self):
        bin_groups = {}
        for result in self.history:
            bin_name = result["bin"]
            if bin_name not in bin_groups:
                bin_groups[bin_name] = []
            # Save a clean string showing item name and confidence level
            bin_groups[bin_name].append(f"{result['category']} ({result['confidence']:.2f})")

        print("\n--- Detailed Sorting Summary ---")
        for bin_name in sorted(bin_groups.keys()):
            items_list = bin_groups[bin_name]
            print(f"\n{bin_name} ({len(items_list)} items total):")
            print("  -> " + ", ".join(items_list))

# initialize report
sorting_report = Report()

# Initialize sorting report
sorter = Sorter()

# Initialize the webcam and set resolution to 640x480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("Webcam opened successfully. Processing waste items... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break
        
    # Run YOLOv8 inference filtered strictly for target waste IDs
    results = model(frame, classes=WASTE_CLASS_IDS, stream=True)
    
    detected_items = []
    annotated_frame = frame.copy()
    for result in results:
        annotated_frame = result.plot()
        
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            category = model.names[class_id]
            score = float(box.conf)

            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            location = (int(x1), int(y1), int(x2), int(y2))

            # Instantiate WasteItem metadata object
            item_obj = WasteItem(category, score, location)
            detected_items.append(item_obj)

            # Sorter applies the rules + threshold and decides the bin
            assigned_bin, reason = sorter.sort(item_obj)

            # Record the result in the report
            sorting_report.add_result(
                item=item_obj,
                category=category,
                confidence=round(score, 2),
                bin_name=assigned_bin
            )
    # Output live detections on console
    if detected_items:
        print(f"Frame Analysis -> Found {len(detected_items)} item(s): " + 
              ", ".join([f"{item.category} ({item.score:.2f})" for item in detected_items]))
        
    cv2.imshow('YOLOv8 Waste Detection Tracker', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""Module 3: Sorter - rules + confidence threshold + bin counters."""

# Cleanup video capture contexts
cap.release()
cv2.destroyAllWindows()

# Display terminal diagnostics after exiting window
sorting_report.show_history()
sorting_report.show_summary() 