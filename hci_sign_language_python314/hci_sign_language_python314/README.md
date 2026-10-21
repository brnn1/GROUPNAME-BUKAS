# HCI & Sign Language — Python 3.14 compatible starter

This version avoids MediaPipe and TensorFlow because package support can vary on Python 3.14. It uses OpenCV's webcam capture and basic skin-color contour segmentation.

## Features
- Desktop GUI with Tkinter
- Webcam preview and approximate hand-contour detection
- Approximate gesture labels and suggested actions

## Important limitations
This is a starter prototype, not a complete sign-language translator. Sign Language mode is explicitly marked as not trained. The basic color-segmentation method is sensitive to lighting, background, and skin tone. Suggested gesture actions are displayed but do not send keyboard shortcuts or control other apps.

## Run in VS Code on Windows
Open this folder in VS Code and select Terminal > New Terminal. Run each command separately:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

If PowerShell blocks environment activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Test
1. Start the application with `python app.py`.
2. Click **Start Camera**.
3. Select **Gesture Control**.
4. Keep your hand inside the central outlined region with good lighting.
5. Try open palm, fist, pointing, and peace sign.

If the webcam does not open, close other apps using the camera and check Windows Settings > Privacy & security > Camera.
