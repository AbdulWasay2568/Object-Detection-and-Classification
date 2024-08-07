import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to load the image from the specified path
def load_image(image_path):
    return cv2.imread(image_path)

# Function to perform detection and classification using YOLO
def detection(image, model_path):
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Perform inference to detect objects
    results = model(image)

    # Define class names based on your model's training data

    # class_names = ["Buffalo", "Elephant", "Rhino", "Zebra"]
    class_names = ["aeorplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]  # Modify based on your actual class names

    # Process the results based on the expected structure
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            if class_id < len(class_names):
                class_name = class_names[class_id]
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, f'{class_name}: {confidence:.2f}', (x_min, y_min - 10), 
                            # cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image

# Function to open file dialog and select an image
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        global input_image_path, original_image
        input_image_path = file_path
        original_image = load_image(input_image_path)
        display_image(original_image, ax, fig)

# Function to display an image using matplotlib
def display_image(image, axis, figure):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axis.clear()
    axis.imshow(image_rgb)
    axis.axis('off')
    figure.canvas.draw()

# Function to handle the detection button click event
def handle_detection():
    if input_image_path:
        detected_image = detection(original_image.copy(), model_path)
        display_image(detected_image, ax, fig)

# Initialize tkinter window
root = tk.Tk()
root.title("Object Detection GUI")

# Set the model path with best.pt
# example 

model_path = r'\runs\detect\LVIS\weights\best.pt' # change the path accordingly

input_image_path = ""
original_image = None

# Create matplotlib figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Embed matplotlib figure in tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Create buttons for input and detection
input_button = tk.Button(root, text="Input Image", command=select_image, font=("Times New Roman", 12, "bold"))
input_button.pack(side=tk.LEFT, padx=10, pady=10)

detect_button = tk.Button(root, text="Detect", command=handle_detection, font=("Times New Roman", 12, "bold"))
detect_button.pack(side=tk.LEFT, padx=10, pady=10)

# Run the tkinter event loop
root.mainloop()
