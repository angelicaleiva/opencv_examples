# OpenCV Tracking Examples

## Description
This repository contains Python scripts that demonstrate object tracking in real-time video.Perfect for those learning computer vision or in need of quick starter code for larger projects. 

It consists of two main scripts:

color_tracker.py: This script tracks objects based on their color using OpenCV.
object_tracker.py: This script uses the YOLO5 model to perform object tracking.


## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Comparison: Color Tracker vs YOLO Object Tracker](#comparison-color-tracker-vs-yolo-object-tracker)
- [Running Unit Tests](#running-unit-tests)
- [Contribution](#contribution)
- [License](#license)
- [Contact](#contact)


## Prerequisites
- Python 3.x
- OpenCV
- PyTorch


## Installation
1. Clone the repository
2. Install dependencies using `pip install -r requirements.txt`

## Usage

Navigate to the `src/` directory:

```bash
cd src/
```

To perform color tracking:
```python
python color_tracker.py
```
To perform object tracking:
```python
python object_tracker.py
```

## Comparison: Color Tracker vs YOLO Object Tracker

This repository contains two different methods for object tracking:

| Method             | Technique                    | Advantages                                          | Disadvantages                                       | Average Processing Time (ms) |
|--------------------|------------------------------|------------------------------------------------------|------------------------------------------------------|-----------------------------|
| **Color Tracker**  | Uses simple OpenCV functions | Fast, lightweight, easy to understand               | Less accurate, works best under controlled lighting | 0.99                        |
| **YOLO Object Tracker** | Uses the YOLO5 model   | Highly accurate, capable of detecting multiple object classes | Slower, requires more computational resources  | 5.24                        |



## Running Unit Tests

To run the unit tests, execute the following command from the root directory of the project:

```bash
python -m unittest tests/test_color_classes.py
```

## Contribution
To contribute, please fork the repository and create a Pull Request.

## License
This project is under the MIT License.

## Contact
For more information, you can contact me at [angelica.leiva@gmail.com](mailto:angelica.leiva@gmail.com)
