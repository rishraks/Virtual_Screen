# Virtual_Screen

Virtual Screen enables users to write on their screen using hand gestures tracked by a webcam. With real-time hand detection and motion tracking, users can draw, annotate, or interact with the screen without physical input devices, offering a seamless and intuitive virtual writing experience.

## Features

- **Real-Time Hand Tracking**: Detects the user's hand and tracks the index finger.
- **Smooth Drawing**: Implements position smoothing for stable and seamless drawing.
- **Boundary Constraint**: Ensures drawing is allowed only within a predefined rectangle.
- **Customizable Canvas**: Clear the canvas using a key press (`c`).
- **Live Feedback**: Displays the tracked finger position and boundary area.

---

## Technologies Used

- **Python 3.7+**
- **OpenCV**: For video capture and frame processing.
- **MediaPipe**: For hand and finger tracking.
- **NumPy**: For efficient array operations.

---

## Setup Instructions

1. **Clone the Repository**

```bash
   git clone https://github.com/rishraks/Virtual_Screen.git
   cd virtual-drawing-canvas
```

2. **Install Dependencies Use pip to install the required libraries**

```bash
pip install opencv-python mediapipe numpy
```

3. **Run the Application Execute the Python script**

```bash
python Virtual_Screen.py
```

## How to Use

1. Start Drawing:

- Position your index finger within the boundary rectangle displayed on the screen.
- Move your finger to draw lines on the virtual canvas.

2. Clear Canvas:

- Press the c key to clear the canvas.

3. Exit:

- Press the q key to quit the application.



## Future Enhancements
- Add support for multiple colors and brush sizes.
- Save the canvas as an image file.
- Implement multi-finger gesture recognition for advanced controls.


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- MediaPipe for robust hand tracking.
- OpenCV for frame processing.
- Community tutorials and resources on computer vision.