# Two Way Sign Language Translator (Windows)

This project provides:
- Text to Sign: type text and see sign GIF output
- Sign to Text: webcam-based recognition using MediaPipe

## Requirements
- Windows 10/11
- Python 3.10 (recommended)
- Webcam (for Sign to Text)

## Setup

```powershell
cd "C:\Users\Utkarsh\Desktop\major project\sign language detection system\1"
py -3.10 -m venv .venv310
.\.venv310\Scripts\activate.bat
```

Install dependencies:

```powershell
.\.venv310\Scripts\python.exe -m pip install -r requirements-run.txt
```

## Configure asset paths (important)
Update these two lines in `main1.py` to point to your local folders:

```python
op_dest = "C:/Users/Utkarsh/Desktop/major project/sign language detection system/1/filtered_data/"
alpha_dest = "C:/Users/Utkarsh/Desktop/major project/sign language detection system/1/alphabet/"
```

## Run

```powershell
.\.venv310\Scripts\python.exe main1.py
```

From the UI:
- Text to Sign: enter text and click Convert
- Sign to Text: click Start Video and use your webcam (close the webcam window to stop)

## Files
- `main1.py`: GUI app (text-to-sign + sign-to-text)
- `models/`: pretrained model weights
- `alphabet/` and `filtered_data/`: assets for sign GIF generation

## Troubleshooting
- If the webcam window won’t close, close the OpenCV window directly (X) or press `q`.
- If MediaPipe errors appear, reinstall with the pinned version in `requirements-run.txt`.
