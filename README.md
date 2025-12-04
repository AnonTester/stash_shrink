# Stash Shrink

![Stash Shrink Logo](static/logo.png) 

**Stash Shrink** is a web app that lets you find video files in stash based on specific properties and convert them using ffmpeg to save disk space.

You can find all video files with bitrate, framerate, width and height over your threshold, find all videos not in your preferred format or with a particular path/filename.

The app converts into x264 codec using CPU only to get the best quality and compression. Desired output properties can be set.

This app started as a collection of bash scripts, so the first python version is v2.0. 

---

## 🌟 Overview

Stash Shrink runs a small web server (default port **9899**) that gives you a search form and task queue to manage the conversions.

Main features:

- Web UI for searching files, managing and monitoring conversions
- Real-time progress updates (ETA, percentage)
- Cancel, resume, or retry failed conversions
- Maximum concurrent conversion setting
- Path mapping configuration for docker or when using on a different system

---

## ⚙️ Installation

### 1. Clone or download Stash Shrink

```bash
git clone https://github.com/AnonTester/stash_shrink.git
cd stash_shrink
```

### 2. Install requirements

You need **Python 3.9+**.

You will need to have ffmpeg installed:

```bash
sudo apt install ffmpeg
```


It is strongly suggested to use a virtual environment for the requirements

On debian/ubuntu systems install the python-venv package for your python version. For python 3.12:

```bash
sudo apt install python3.12-venv
```

Then create the virtual environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required python packages:
```bash
pip install -r requirements.txt
```

To deactivate the virtual environment:
```bash
deactivate
```


### 3. Start Stash Shrink

enable virtual environment:
```bash
cd stash_shrink
source .venv/bin/activate
```
then start tube-q

```bash
python3 stash_shrink.py
```

Then open your browser and go to:

👉 **http://localhost:9899**

to exit/deactivate the virtual environment:
```bash
deactivate
```

---

## 🧰 Configuration

On first run, the settings page opens by default for you to set stash end point details and your preferences.

By default, converted files are stored in the same directory as the original with added sequential number and then attached the stash scene as additional scene file. Use this to experiment with quality settings. Ony when happy with the result, change the configuration to overwrite the original file.

---

# 🧾 License

MIT License © 2025.
