import os
import json
from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid
import signal
import subprocess
import shutil
import re
import time
import concurrent.futures
from queue import Queue

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version
VERSION = "1.1.0"

# Configuration
CONFIG_FILE = "config.json"
LOGS_DIR = "logs"

# Default configuration
DEFAULT_CONFIG = {
    "stash_url": "http://localhost:9999",
    "api_key": "",
    "overwrite_original": True,
    "default_search_limit": 50,
    "max_concurrent_tasks": 2,
    "video_settings": {
        "width": 1280,
        "height": 720,
        "bitrate": "1000k",
        "framerate": 30,
        "buffer_size": "2000k",
        "container": "mp4",
        "crf": 26  # ADDED: Default CRF value
    },
    "path_mappings": []
}

# Global state
queue_initialized = False
last_sse_data = None
conversion_queue = []
conversion_tasks = {}
active_tasks = set()
QUEUE_STATE_FILE = "conversion_queue.json"
task_status = {}
sse_clients = set()

# Thread pool for FFmpeg processes
ffmpeg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
sse_update_queue = Queue()

# Ensure logs directory exists
Path(LOGS_DIR).mkdir(exist_ok=True)

class Settings(BaseModel):
    stash_url: str
    api_key: str
    default_search_limit: int
    max_concurrent_tasks: int
    overwrite_original: bool = True
    video_settings: Dict[str, Any]
    path_mappings: List[str]

class SearchParams(BaseModel):
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_bitrate: Optional[str] = None
    max_framerate: Optional[float] = None
    codec: Optional[str] = None
    path: Optional[str] = None

class SceneFile(BaseModel):
    id: str
    size: Optional[int] = None
    basename: str
    path: str
    bit_rate: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    frame_rate: Optional[float] = None
    duration: Optional[float] = None
    video_codec: Optional[str] = None

class Scene(BaseModel):
    id: str
    title: str
    details: Optional[str] = None
    files: List[SceneFile]

    @property
    def path(self):
        return self.files[0].path if self.files else ""

    @property
    def file(self):
        if self.files and len(self.files) > 0:
            return {
                'size': self.files[0].size,
                'duration': self.files[0].duration,
                'video_codec': self.files[0].video_codec,
                'width': self.files[0].width,
                'height': self.files[0].height,
                'bit_rate': self.files[0].bit_rate,
                'frame_rate': self.files[0].frame_rate
            }
        return {}

class ConversionTask(BaseModel):
    task_id: str
    scene: Scene
    status: str = "pending"  # pending, processing, completed, completed_with_warning, error, cancelled
    progress: float = 0.0
    eta: Optional[float] = None
    log_file: str
    output_file: Optional[str] = None
    error: Optional[str] = None

class FFmpegProgress:
    def __init__(self, total_duration: float, task_id: str = None):
        self.total_duration = total_duration
        self.current_time = 0.0
        self.progress = 0.0
        self.last_update = time.time()
        self.last_progress_update = 0.0
        self.task_id = task_id or "unknown"
        self.line_count = 0
        self.parsed_count = 0

    def parse_line(self, line: str):
        """Parse FFmpeg output line to extract progress information"""
        self.line_count += 1
        line = line.strip()

        current_time = time.time()

        # Only process progress updates every 0.2 seconds to reduce load
        time_since_last_update = current_time - self.last_update
        if time_since_last_update < 0.2:
            return False

        self.last_update = current_time
        updated = False

        # Debug: Log first few lines to see what we're getting
        if self.line_count <= 20:  # First 20 lines
            logger.debug(f"[Task {self.task_id}] Line {self.line_count}: {line[:200]}")

        # Parse standard FFmpeg output format:
        # frame= 1030 fps=229 q=33.0 size=    3584KiB time=00:00:41.12 bitrate= 714.0kbits/s speed=9.14x
        # OR the multi-line version:
        # frame=  1234
        # fps=  229
        # q=33.0
        # size=    3584KiB
        # time=00:00:41.12
        # bitrate= 714.0kbits/s
        # speed=9.14x

        # First, try to parse time from the line
        time_match = None

        # Pattern 1: time=00:00:41.12 (with optional spaces)
        time_patterns = [
            r'time=(\d+:\d+:\d+\.\d+)',  # time=00:00:41.12
            r'time=(\d{2}:\d{2}:\d{2}\.\d{2})',  # time=00:00:41.12
            r'time=(\d{2}:\d{2}:\d{2})',  # time=00:00:41
        ]

        for pattern in time_patterns:
            match = re.search(pattern, line)
            if match:
                time_match = match
                break

        if time_match:
            time_str = time_match.group(1)
            self.current_time = self.parse_time_string(time_str)
            if self.total_duration > 0:
                new_progress = min((self.current_time / self.total_duration) * 100, 99.0)
                if abs(new_progress - self.progress) > 0.1:  # More than 0.1% change
                    self.progress = new_progress
                    updated = True
                    self.parsed_count += 1
                    logger.debug(f"[Task {self.task_id}] Progress from time: {self.progress:.1f}% (time: {time_str})")

        # If no time found, try to parse frame number
        elif 'frame=' in line:
            frame_match = re.search(r'frame=\s*(\d+)', line)
            if frame_match:
                try:
                    frame_count = int(frame_match.group(1))
                    # Estimate progress based on frames (assuming source framerate or default 30)
                    if self.total_duration > 0:
                        # Try to get frame rate from the line too
                        fps_match = re.search(r'fps=\s*([\d\.]+)', line)
                        fps = 30.0  # default
                        if fps_match:
                            try:
                                fps = float(fps_match.group(1))
                            except ValueError:
                                pass

                        estimated_duration = frame_count / fps if fps > 0 else frame_count / 30.0
                        new_progress = min((estimated_duration / self.total_duration) * 100, 99.0)
                        if abs(new_progress - self.progress) > 0.1:
                            self.progress = new_progress
                            updated = True
                            self.parsed_count += 1
                            logger.debug(f"[Task {self.task_id}] Progress from frames: {self.progress:.1f}% (frame: {frame_count}, fps: {fps})")
                except (ValueError, IndexError) as e:
                    logger.debug(f"[Task {self.task_id}] Failed to parse frame: {e}")

        return updated

    def parse_time_string(self, time_str: str) -> float:
        """Convert time string (HH:MM:SS.ms) to seconds"""
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError):
            pass
        return 0.0

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            if 'delete_original' in config and 'overwrite_original' not in config:
                config['overwrite_original'] = config['delete_original']
                del config['delete_original']
                save_config(config)
            return config
    return None

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def get_config():
    config = load_config()
    if config is None:
        return DEFAULT_CONFIG
    return config

def is_first_run():
    return not os.path.exists(CONFIG_FILE)

def save_queue_state():
    """Save the conversion queue to disk"""
    try:
        queue_data = []
        for task in conversion_queue:
            task_data = {
                "task_id": task.task_id,
                "scene": task.scene.dict(),
                "status": task.status,
                "progress": task.progress,
                "eta": task.eta,
                "log_file": task.log_file,
                "output_file": task.output_file,
                "error": task.error
            }
            queue_data.append(task_data)

        with open(QUEUE_STATE_FILE, 'w') as f:
            json.dump(queue_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save queue state to {QUEUE_STATE_FILE}: {e}")

def load_queue_state():
    """Load the conversion queue from disk"""
    global conversion_queue
    global task_status

    conversion_queue.clear()
    task_status.clear()

    if os.path.exists(QUEUE_STATE_FILE):
        try:
            logger.info(f"Loading queue state from {QUEUE_STATE_FILE}")
            with open(QUEUE_STATE_FILE, 'r') as f:
                queue_data = json.load(f)

            for task_data in queue_data:
                scene = Scene(**task_data["scene"])
                task = ConversionTask(
                    task_id=task_data["task_id"],
                    scene=scene,
                    status=task_data["status"],
                    progress=task_data["progress"],
                    eta=task_data["eta"],
                    log_file=task_data["log_file"],
                    output_file=task_data["output_file"],
                    error=task_data["error"]
                )
                conversion_queue.append(task)

            logger.info(f"Loaded {len(conversion_queue)} tasks from queue state")
        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")

def initialize_queue_system():
    """Initialize the queue system - called once on app startup"""
    load_queue_state()
    logger.info(f"Queue system initialized with {len(conversion_queue)} tasks")

def apply_path_mappings(file_path: str, path_mappings: List[str]) -> str:
    """
    Apply path mappings to convert between Docker and host paths.
    mappings format: ["docker_path:host_path", ...]
    """
    if not path_mappings:
        return file_path

    for mapping in path_mappings:
        if ':' in mapping:
            docker_path, host_path = mapping.split(':', 1)
            docker_path = docker_path.strip()
            host_path = host_path.strip()

            docker_path_norm = docker_path.rstrip('/') + '/'
            host_path_norm = host_path.rstrip('/') + '/'

            if file_path.startswith(docker_path_norm):
                return file_path.replace(docker_path_norm, host_path_norm, 1)
            elif file_path.startswith(host_path_norm):
                return file_path.replace(host_path_norm, docker_path_norm, 1)

    return file_path

async def stash_request(graphql_query: str, variables: dict = None):
    config = get_config()
    headers = {}
    if config.get('api_key'):
        headers['ApiKey'] = config['api_key']

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Making GraphQL request to {config['stash_url']}/graphql")
            logger.debug(f"GraphQL Query: {graphql_query}")
            if variables:
                logger.debug(f"GraphQL Variables: {variables}")

            response = await client.post(
                f"{config['stash_url']}/graphql",
                json={"query": graphql_query, "variables": variables},
                headers=headers
            )

            logger.debug(f"Stash response status: {response.status_code}")
            if response.status_code != 200:
                logger.debug(f"Stash response body: {response.text}")

            response.raise_for_status()
            result = response.json()

            if 'data' in result and 'findScenes' in result['data']:
                scenes_count = len(result['data']['findScenes'].get('scenes', []))
                reported_count = result['data']['findScenes'].get('count', 0)
                logger.debug(f"Stash response: {scenes_count} scenes returned, {reported_count} reported total")

            return result
    except httpx.ConnectError as e:
        logger.error(f"Connection error to Stash at {config['stash_url']}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Cannot connect to Stash at {config['stash_url']}. Please check if the URL is correct and Stash is running."
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to Stash: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Connection timeout to Stash at {config['stash_url']}. Please check if Stash is running and accessible."
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Stash: {e.response.status_code}")
        error_detail = f"Stash returned error {e.response.status_code}"
        try:
            error_data = e.response.json()
            if 'errors' in error_data:
                error_messages = [err.get('message', 'Unknown error') for err in error_data['errors']]
                error_detail += f": {', '.join(error_messages)}"
        except:
            pass
        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        logger.error(f"Unexpected error during Stash request: {e}")
        raise HTTPException(status_code=400, detail=f"Unexpected error contacting Stash: {str(e)}")

def run_ffmpeg_with_progress(task, ffmpeg_cmd, temp_output, file_duration, start_time):
    """Run FFmpeg and update progress (run in thread pool)"""
    # Check if temp file already exists (from previous attempt)
    if os.path.exists(temp_output):
        logger.info(f"[Task {task.task_id}] Temp file exists from previous attempt: {temp_output}")
        # Check if it's a valid video file by size
        if os.path.getsize(temp_output) > 0:
            logger.info(f"[Task {task.task_id}] Temp file has content, checking if conversion should resume...")
            # For now, we'll overwrite it. In a more advanced version, we could check if we can resume.
            # FFmpeg doesn't easily support resuming, so we'll delete and start over.
            os.remove(temp_output)
            logger.info(f"[Task {task.task_id}] Deleted existing temp file to start fresh")

    progress_tracker = FFmpegProgress(file_duration, task.task_id)

    # Run FFmpeg with line buffering
    process = subprocess.Popen(
        ffmpeg_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        bufsize=1,  # Line buffered
        universal_newlines=True,
        text=True
    )

    # Read output line by line
    last_sse_update = start_time
    sse_update_interval = 2.0

    while True:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.1)
            continue

        line = line.strip()
        if not line:
            continue

        # Write to log with timestamp
        current_time = time.time()
        with open(task.log_file, 'a') as log:
            log.write(f"[{current_time:.3f}] {line}\n")

        # Parse progress
        if progress_tracker.parse_line(line):
            task.progress = progress_tracker.progress

            # Calculate ETA
            if task.progress > 0:
                elapsed_time = current_time - start_time
                if task.progress < 100:
                    total_estimated_time = elapsed_time / (task.progress / 100)
                    remaining_time = total_estimated_time - elapsed_time
                    task.eta = max(0, remaining_time)
                else:
                    task.eta = 0

            # Request SSE update via queue
            if current_time - last_sse_update >= sse_update_interval:
                sse_update_queue.put("update")
                last_sse_update = current_time

        # Check for cancellation
        if task.status == "cancelled":
            process.terminate()
            break

    return process.wait()

async def convert_video_threaded(task: ConversionTask):
    """Run FFmpeg in a thread pool to avoid blocking the event loop"""
    config = get_config()
    overwrite_original = config.get('overwrite_original', True)
    video_settings = config['video_settings']

    try:
        if not task.scene.files or len(task.scene.files) == 0:
            raise Exception("No files found in scene")

        scene_file = task.scene.files[0]
        input_file = scene_file.path

        original_extension = os.path.splitext(input_file)[1].lower()
        new_extension = f".{video_settings['container']}"

        # Determine output filename
        if overwrite_original and original_extension == new_extension:
            base_name = os.path.splitext(input_file)[0]
            final_output = input_file
            temp_output = f"{base_name}.converting.{video_settings['container']}"
        elif not overwrite_original:
            base_name = os.path.splitext(input_file)[0]
            final_output = await find_available_filename(base_name, video_settings['container'])
            temp_output = f"{final_output}.converting"
        else:
            base_name = os.path.splitext(input_file)[0]
            final_output = await find_available_filename(base_name, video_settings['container'])
            temp_output = f"{final_output}.converting"

        # Get file duration for progress calculation
        file_duration = scene_file.duration or 0

        logger.info(f"[Task {task.task_id}] Starting conversion for {input_file}")

        # Build FFmpeg command
        framerate_option = ""
        if video_settings.get('framerate'):
            if scene_file.frame_rate and scene_file.frame_rate > video_settings['framerate']:
                framerate_option = f"-r {video_settings['framerate']}"

        # Simple FFmpeg command
        ffmpeg_cmd = f"""ffmpeg -y -hide_banner -stats_period 0.5 -i "{input_file}" -filter_complex "scale=ceil(iw*min(1\,min({video_settings['width']}/iw\,{video_settings['height']}/ih))/2)*2:-2" -c:v libx264 {framerate_option} -crf {video_settings.get('crf', 26)} -c:a aac -b:v {video_settings['bitrate']} -maxrate {video_settings['bitrate']} -bufsize {video_settings['buffer_size']} -f {video_settings['container']} "{temp_output}" """

        logger.debug(f"[Task {task.task_id}] FFmpeg command: {ffmpeg_cmd}")

        start_time = time.time()

        # Check if this is a retry and if the output file already exists
        is_retry = task.status in ["error", "cancelled"]
        output_exists = os.path.exists(final_output) or os.path.exists(temp_output)

        if is_retry and output_exists:
            logger.info(f"[Task {task.task_id}] Retry detected, checking for existing output files...")

            # Check which files exist
            final_exists = os.path.exists(final_output)
            temp_exists = os.path.exists(temp_output)

            if final_exists:
                # Final output exists - conversion was successful, only Stash update failed
                logger.info(f"[Task {task.task_id}] Final output already exists: {final_output}")
                logger.info(f"[Task {task.task_id}] Assuming conversion succeeded, resuming from Stash update")

                # Skip FFmpeg conversion, just do Stash updates
                returncode = 0
                task.progress = 100.0
                task.eta = 0

                # Write to log
                with open(task.log_file, 'a') as log:
                    log.write(f"--- RETRY DETECTED ---\n")
                    log.write(f"Final output already exists: {final_output}\n")
                    log.write(f"Skipping conversion, proceeding directly to Stash update\n")
                    log.write("-" * 80 + "\n")

            elif temp_exists:
                # Temp file exists - conversion was interrupted
                logger.info(f"[Task {task.task_id}] Temp file exists: {temp_output}")
                logger.info(f"[Task {task.task_id}] Resuming interrupted conversion")

                # We'll run FFmpeg normally, it will overwrite the temp file
                with open(task.log_file, 'a') as log:
                    log.write(f"--- RETRY DETECTED ---\n")
                    log.write(f"Temp file exists: {temp_output}\n")
                    log.write(f"Resuming conversion\n")
                    log.write("-" * 80 + "\n")

                # Continue with normal conversion
                returncode = await run_ffmpeg_conversion(task, ffmpeg_cmd, temp_output, file_duration)

            else:
                # Should not happen since output_exists was True
                returncode = await run_ffmpeg_conversion(task, ffmpeg_cmd, temp_output, file_duration)
        else:
            # First attempt or no existing output files
            logger.info(f"[Task {task.task_id}] Starting conversion for {input_file}")

            # Write to log
            with open(task.log_file, 'a') as log:
                log.write(f"Starting conversion: {input_file} -> {final_output}\n")
                log.write(f"File duration: {file_duration} seconds\n")
                log.write(f"FFmpeg command: {ffmpeg_cmd}\n")
                log.write("-" * 80 + "\n")

            returncode = await run_ffmpeg_conversion(task, ffmpeg_cmd, temp_output, file_duration)

        logger.info(f"[Task {task.task_id}] FFmpeg process completed with return code: {returncode}")

        # Final log entry
        with open(task.log_file, 'a') as log:
            log.write("-" * 80 + "\n")
            log.write(f"FFmpeg process completed with return code: {returncode}\n")

        # Check if task was cancelled before processing result
        if task.status == "cancelled":
            logger.info(f"[Task {task.task_id}] Task was cancelled, skipping result processing")
            if 'temp_output' in locals() and os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                    logger.debug(f"[Task {task.task_id}] Cleaned up temp file for cancelled task: {temp_output}")
                except Exception as cleanup_error:
                    logger.error(f"[Task {task.task_id}] Failed to clean up temp file: {cleanup_error}")
            save_queue_state()
            clear_sse_cache()
            return

        if returncode == 0:
            # Verify output file exists (either newly created or existing from previous run)
            if not os.path.exists(temp_output) and not os.path.exists(final_output):
                raise Exception(f"Output file was not created: {temp_output} or {final_output}")

            # If temp file exists but final doesn't, rename it
            if os.path.exists(temp_output) and not os.path.exists(final_output):
                output_size = os.path.getsize(temp_output)
                if output_size == 0:
                    raise Exception(f"Output file is empty: {temp_output}")

                logger.debug(f"[Task {task.task_id}] Renaming temp file to final: {temp_output} -> {final_output}")
                os.rename(temp_output, final_output)

            # Verify final output
            if not os.path.exists(final_output):
                raise Exception(f"Final output file does not exist: {final_output}")

            output_size = os.path.getsize(final_output)
            logger.debug(f"[Task {task.task_id}] Final output file: {final_output} ({output_size} bytes)")

            # Handle file operations based on settings
            try:
                if overwrite_original and original_extension == new_extension:
                    # Already handled during conversion (original replaced)
                    await trigger_stash_scan(final_output)
                elif overwrite_original and original_extension != new_extension:
                    await update_stash_file(task.scene.id, scene_file.id, final_output, overwrite_original)

                    # Only delete original after successful Stash update
                    if overwrite_original and os.path.exists(input_file):
                        os.remove(input_file)
                        logger.info(f"[Task {task.task_id}] Deleted original file: {input_file}")
                else:
                    # Non-overwrite mode: add as new file
                    await add_file_to_scene(task.scene.id, final_output, overwrite_original)

                task.status = "completed"
                task.output_file = final_output
                task.progress = 100.0
                task.eta = 0
                logger.info(f"[Task {task.task_id}] Conversion completed successfully")

            except Exception as stash_error:
                # Stash update failed, but conversion succeeded
                # Don't mark as error if the file was created successfully
                # User can manually fix Stash issues
                task.status = "completed_with_warning"
                task.output_file = final_output
                task.progress = 100.0
                task.eta = 0
                task.error = f"Conversion succeeded but Stash update failed: {str(stash_error)}"
                logger.warning(f"[Task {task.task_id}] Conversion succeeded but Stash update failed: {stash_error}")

            save_queue_state()
            clear_sse_cache()
            logger.debug(f"[Task {task.task_id}] Queue state saved and SSE cache cleared")

        else:
            # Check if task was cancelled during processing
            if task.status != "cancelled":
                task.status = "error"
                task.error = f"FFmpeg failed with return code {returncode}"
                logger.error(f"[Task {task.task_id}] FFmpeg conversion failed for {input_file}")
                logger.error(f"[Task {task.task_id}] Error details saved to: {task.log_file}")
            else:
                logger.info(f"[Task {task.task_id}] Task cancelled, FFmpeg return code: {returncode}")

            save_queue_state()
            clear_sse_cache()

    except Exception as e:
        # Only set to error if not already cancelled
        if task.status != "cancelled":
            task.status = "error"
            task.error = str(e)
        logger.error(f"[Task {task.task_id}] Conversion failed: {e}", exc_info=True)

        # Clean up temp file if exists
        if 'temp_output' in locals() and os.path.exists(temp_output):
            try:
                os.remove(temp_output)
                logger.debug(f"[Task {task.task_id}] Cleaned up temp file after error: {temp_output}")
            except Exception as cleanup_error:
                logger.error(f"[Task {task.task_id}] Failed to clean up temp file: {cleanup_error}")

    finally:
        # Remove from active tasks
        if task.task_id in active_tasks:
            active_tasks.remove(task.task_id)
            logger.debug(f"[Task {task.task_id}] Removed from active tasks")

        # Force SSE update
        logger.debug(f"[Task {task.task_id}] Forcing final SSE update")
        clear_sse_cache()

        # Start next task in queue if not paused
        if not queue_paused:
            logger.debug(f"[Task {task.task_id}] Queue not paused, checking for next task")
            asyncio.create_task(process_conversion_queue())



async def run_ffmpeg_conversion(task, ffmpeg_cmd, temp_output, file_duration):
    """Run FFmpeg conversion and return the return code"""
    start_time = time.time()

    # Run FFmpeg in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        ffmpeg_executor,
        run_ffmpeg_with_progress,
        task,
        ffmpeg_cmd,
        temp_output,
        file_duration,
        start_time
    )

async def process_sse_update_queue():
    """Background task to process SSE update requests from threads"""
    while True:
        try:
            # Check if there are any update requests
            try:
                while not sse_update_queue.empty():
                    sse_update_queue.get_nowait()
                    logger.debug("Processing SSE update request from thread")
                    clear_sse_cache()
            except:
                pass

            # Sleep for a short time
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in SSE update queue processor: {e}")
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Stash Shrink")
    initialize_queue_system()
    # Queue starts paused by default
    global queue_paused
    queue_paused = True

    # Start SSE update processor
    sse_processor_task = asyncio.create_task(process_sse_update_queue())

    yield

    # Shutdown
    logger.info("Shutting down Stash Shrink")
    # Cancel the SSE processor task
    sse_processor_task.cancel()
    try:
        await sse_processor_task
    except asyncio.CancelledError:
        pass

    # Clean up any active tasks
    for task_id in list(active_tasks):
        task = task_status.get(task_id)
        if task and task.status == "processing":
            task.status = "pending"
            logger.info(f"Reset active task {task_id} to pending on shutdown")
    save_queue_state()

# Create FastAPI app with lifespan
app = FastAPI(title="Stash Shrink", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Runtime queue state (not persisted)
queue_paused = True

def clear_sse_cache():
    global last_sse_data
    last_sse_data = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    config = get_config()
    show_settings = is_first_run()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": config,
        "show_settings": show_settings,
        "version": VERSION
    })

@app.get("/api/config")
async def get_config_api():
    config = get_config()
    return config

@app.post("/api/config")
async def update_config(settings: Settings):
    save_config(settings.dict())
    return {"status": "ok"}

@app.post("/api/search")
async def search_scenes(search_params: SearchParams):
    try:
        query = """
        query FindAllScenes {
          findScenes(
            filter: {
              per_page: -1
            }
          ) {
            count
            scenes {
              id
              title
              details
              files {
                id
                size
                basename
                path
                bit_rate
                height
                width
                frame_rate
                duration
                video_codec
              }
            }
          }
        }
        """

        logger.info("Executing GraphQL query to find all scenes")
        result = await stash_request(query)

        if 'errors' in result:
            query_fallback = """
            query FindAllScenes {
              findScenes(
                filter: {
                  per_page: 10000
                }
              ) {
                count
                scenes {
                  id
                  title
                  details
                  files {
                    id
                    size
                    basename
                    path
                    bit_rate
                    height
                    width
                    frame_rate
                    duration
                    video_codec
                  }
                }
              }
            }
            """
            logger.info("Trying fallback query with per_page: 10000")
            result = await stash_request(query_fallback)

            if 'errors' in result:
                raise HTTPException(status_code=400, detail=f"GraphQL error: {result['errors']}")

        scenes_data = result['data']['findScenes']['scenes']
        actual_count = len(scenes_data)

        logger.info(f"Found {actual_count} scenes in Stash")

        scenes = []
        config = get_config()
        path_mappings = config.get('path_mappings', [])

        for scene_data in scenes_data:
            try:
                if scene_data.get('files'):
                    files = []
                    for file_data in scene_data['files']:
                        try:
                            processed_file_data = {
                                'id': file_data.get('id', ''),
                                'size': file_data.get('size', 0),
                                'basename': file_data.get('basename', ''),
                                'path': file_data.get('path', ''),
                                'bit_rate': file_data.get('bit_rate', 0),
                                'height': file_data.get('height', 0),
                                'width': file_data.get('width', 0),
                                'frame_rate': file_data.get('frame_rate', 0),
                                'duration': file_data.get('duration', 0),
                                'video_codec': file_data.get('video_codec', '')
                            }

                            if processed_file_data['path']:
                                processed_file_data['path'] = apply_path_mappings(
                                    processed_file_data['path'], path_mappings
                                )

                            scene_file = SceneFile(**processed_file_data)
                            files.append(scene_file)
                        except Exception as e:
                            logger.warning(f"Failed to parse file data: {file_data}, error: {e}")
                            continue

                    filtered_files = []
                    for file in files:
                        include_file = False
                        exceeds_limits = False
                        wrong_codec = False
                        path_matches = True

                        # Check width
                        if search_params.max_width is not None and file.width is not None:
                            if file.width > search_params.max_width:
                                exceeds_limits = True

                        # Check height
                        if search_params.max_height is not None and file.height is not None:
                            if file.height > search_params.max_height:
                                exceeds_limits = True

                        # Check bitrate
                        if search_params.max_bitrate and file.bit_rate:
                            bitrate_value = convert_bitrate_to_bps(search_params.max_bitrate)
                            if file.bit_rate > bitrate_value:
                                exceeds_limits = True

                        # Check framerate
                        if search_params.max_framerate is not None and file.frame_rate is not None:
                            if file.frame_rate > search_params.max_framerate:
                                exceeds_limits = True

                        # Check codec
                        if search_params.codec and file.video_codec:
                            file_codec = (file.video_codec or '').lower().replace('.', '')
                            search_codec = search_params.codec.lower().replace('.', '')
                            if file_codec != search_codec:
                                wrong_codec = True

                        # Check path filter
                        if search_params.path:
                            search_path_lower = search_params.path.lower()
                            file_path_lower = file.path.lower()
                            if search_path_lower not in file_path_lower:
                                path_matches = False

                        has_technical_filters = any([
                            search_params.max_width is not None,
                            search_params.max_height is not None,
                            search_params.max_bitrate is not None,
                            search_params.max_framerate is not None,
                            search_params.codec is not None
                        ])

                        if not any([
                            search_params.max_width is not None,
                            search_params.max_height is not None,
                            search_params.max_bitrate is not None,
                            search_params.max_framerate is not None,
                            search_params.codec is not None,
                            search_params.path is not None
                        ]):
                            include_file = True
                        elif has_technical_filters:
                            if (exceeds_limits or wrong_codec) and path_matches:
                                include_file = True
                        else:
                            if path_matches:
                                include_file = True

                        if include_file:
                            filtered_files.append(file)

                    if filtered_files:
                        scene_data['files'] = filtered_files
                        scene = Scene(**scene_data)
                        scenes.append(scene)

            except Exception as e:
                logger.error(f"Failed to process scene data: {scene_data}, error: {e}")
                continue

        logger.info(f"Returning {len(scenes)} scenes after filtering")
        return {"scenes": scenes}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search_scenes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during search: {str(e)}")

@app.get("/api/conversion-log/{task_id}")
async def get_conversion_log(task_id: str):
    try:
        task = next((t for t in conversion_queue if t.task_id == task_id), None)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if os.path.exists(task.log_file):
            with open(task.log_file, 'r') as f:
                log_content = f.read()
            return {"log": log_content}
        else:
            return {"log": "Log file not found"}
    except Exception as e:
        logger.error(f"Failed to read log for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read log: {str(e)}")

@app.post("/api/queue-conversion")
async def queue_conversion(scene_ids: List[str]):
    config = get_config()
    path_mappings = config.get('path_mappings', [])

    existing_scene_ids = {task.scene.id for task in conversion_queue}
    new_scene_ids = set(scene_ids) - existing_scene_ids
    scene_ids = list(new_scene_ids)

    try:
        scenes_query = """
        query GetScenes($ids: [ID!]) {
          findScenes(ids: $ids) {
            scenes {
              id
              title
              details
              files {
                id
                size
                basename
                path
                bit_rate
                height
                width
                frame_rate
                duration
                video_codec
              }
            }
          }
        }
        """

        result = await stash_request(scenes_query, {"ids": scene_ids})
        if 'errors' in result:
            raise HTTPException(status_code=400, detail=result['errors'])

        queued_count = 0
        for scene_data in result['data']['findScenes']['scenes']:
            try:
                if scene_data.get('files') and len(scene_data['files']) > 0:
                    for file_data in scene_data['files']:
                        if 'path' in file_data and file_data['path']:
                            file_data['path'] = apply_path_mappings(
                                file_data['path'], path_mappings
                            )

                    scene = Scene(**scene_data)
                    task_id = str(uuid.uuid4())
                    log_file = os.path.join(LOGS_DIR, f"{scene.id}.log")

                    task = ConversionTask(
                        task_id=task_id,
                        scene=scene,
                        log_file=log_file
                    )

                    conversion_queue.append(task)
                    task_status[task_id] = task
                    queued_count += 1
                    save_queue_state()
            except Exception as e:
                logger.error(f"Failed to queue scene {scene_data.get('id', 'unknown')}: {e}")
                continue

        if queued_count > 0 and len(active_tasks) < config['max_concurrent_tasks']:
            asyncio.create_task(process_conversion_queue())

        return {"status": "queued", "count": queued_count}

    except Exception as e:
        logger.error(f"Error queueing conversion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to queue conversion: {str(e)}")

async def process_conversion_queue():
    config = get_config()

    if queue_paused:
        logger.debug("Queue is paused, not processing")
        return

    pending_tasks = [task for task in conversion_queue if task.status in ["pending"]]
    if not pending_tasks:
        logger.debug("No pending tasks to process")
        return

    available_slots = config['max_concurrent_tasks'] - len(active_tasks)
    if available_slots <= 0:
        logger.debug(f"No available slots (active: {len(active_tasks)}, max: {config['max_concurrent_tasks']})")
        return

    tasks_to_start = min(available_slots, len(pending_tasks))
    logger.info(f"Starting {tasks_to_start} tasks from {len(pending_tasks)} pending tasks")

    for i in range(tasks_to_start):
        task = pending_tasks[i]
        active_tasks.add(task.task_id)
        task.status = "processing"
        asyncio.create_task(convert_video_threaded(task))

        await asyncio.sleep(0.1)

    clear_sse_cache()

async def fix_stash_update(task: ConversionTask):
    """Only fix the Stash update for a task that already has a converted file"""
    config = get_config()
    overwrite_original = config.get('overwrite_original', True)

    try:
        if not task.scene.files or len(task.scene.files) == 0:
            raise Exception("No files found in scene")

        scene_file = task.scene.files[0]
        input_file = scene_file.path
        final_output = task.output_file

        # Double-check that output file still exists (it could have been deleted between check and now)
        if not final_output or not os.path.exists(final_output):
            raise Exception(f"Output file no longer exists: {final_output}")

        logger.info(f"[Fix Stash] Starting Stash fix for task {task.task_id}")
        logger.info(f"[Fix Stash] Input: {input_file}")
        logger.info(f"[Fix Stash] Output: {final_output}")

        # Update task status
        task.status = "processing"
        task.progress = 100.0  # Already converted
        task.eta = 0
        clear_sse_cache()

        # Write to log
        with open(task.log_file, 'a') as log:
            log.write(f"--- FIXING STASH UPDATE ---\n")
            log.write(f"Retrying Stash update only\n")
            log.write(f"Using existing output file: {final_output}\n")
            log.write("-" * 80 + "\n")

        # Verify output file exists and is valid
        if not final_output or not os.path.exists(final_output):
            raise Exception(f"Output file does not exist: {final_output}")

        output_size = os.path.getsize(final_output)
        if output_size == 0:
            raise Exception(f"Output file is empty: {final_output}")

        logger.debug(f"[Fix Stash] Output file verified: {final_output} ({output_size} bytes)")

        # Determine the operation based on original settings
        original_extension = os.path.splitext(input_file)[1].lower()
        video_settings = config['video_settings']
        new_extension = f".{video_settings['container']}"

        # Perform the appropriate Stash operation
        success = False
        error_message = None
        stash_operation = "unknown"

        try:
            if overwrite_original and original_extension == new_extension:
                # Same extension - already replaced, just trigger scan
                stash_operation = "scan"
                await trigger_stash_scan(final_output)
                logger.info(f"[Fix Stash] Triggered metadata scan for: {final_output}")
                success = True

            elif overwrite_original and original_extension != new_extension:
                # Different extensions with overwrite
                stash_operation = "update"
                await update_stash_file(task.scene.id, scene_file.id, final_output, overwrite_original)
                logger.info(f"[Fix Stash] Updated Stash file entry")

                # Delete original only after successful Stash update
                if os.path.exists(input_file):
                    os.remove(input_file)
                    logger.info(f"[Fix Stash] Deleted original file: {input_file}")
                success = True

            else:
                # Non-overwrite mode: add as new file
                stash_operation = "add"
                await add_file_to_scene(task.scene.id, final_output, overwrite_original)
                logger.info(f"[Fix Stash] Added file to scene")
                success = True

        except Exception as e:
            error_message = str(e)
            logger.warning(f"[Fix Stash] Stash operation '{stash_operation}' failed: {error_message}")
            # Don't re-raise, we'll handle it below

        if success:
            # Success!
            task.status = "completed"
            task.error = None
            logger.info(f"[Fix Stash] Stash update completed successfully for task {task.task_id}")
        else:
            # Failed - set back to warning status
            task.status = "completed_with_warning"
            # If the error is about missing file, provide clearer message
            error_detail = error_message or "Unknown error"
            task.error = f"Stash fix failed: {error_detail}"
            logger.warning(f"[Fix Stash] Failed to fix Stash for task {task.task_id}: {error_message}")

            with open(task.log_file, 'a') as log:
                log.write(f"Stash fix failed: {error_message}\n")

        save_queue_state()
        clear_sse_cache()

    except Exception as e:
        # This is for unexpected errors (file not found, etc.)
        task.status = "completed_with_warning"
        task.error = f"Stash fix failed: {str(e)}"
        logger.error(f"[Fix Stash] Unexpected error fixing Stash for task {task.task_id}: {e}", exc_info=True)

        with open(task.log_file, 'a') as log:
            log.write(f"Unexpected error in Stash fix: {str(e)}\n")

        # If the error is specifically about missing file, reset to pending
        if "no longer exists" in str(e) or "does not exist" in str(e):
            task.status = "pending"
            task.error = "Output file missing. Reset to pending for full retry."

        save_queue_state()
        clear_sse_cache()

    finally:
        # Remove from active tasks if it was added
        if task.task_id in active_tasks:
            active_tasks.remove(task.task_id)

        clear_sse_cache()

@app.get("/api/conversion-status")
async def conversion_status():
    global queue_paused
    config = get_config()

    return {
        "queue": [task.dict() for task in conversion_queue],
        "active": list(active_tasks),
        "completed": [task.dict() for task in conversion_queue if task.status in ["completed", "error"]],
        "paused": config.get('paused', True)
    }

@app.post("/api/cancel-conversion/{task_id}")
async def cancel_conversion(task_id: str):
    for i, task in enumerate(conversion_queue):
        if task.task_id == task_id:
            # Only allow cancelling of pending or processing tasks
            if task.status not in ["pending", "processing"]:
                if task.status == "cancelled":
                    logger.info(f"Task {task_id} is already cancelled")
                    return {"status": "already_cancelled"}
                else:
                    logger.info(f"Task {task_id} has status {task.status}, not cancelling")
                    return {"status": "not_cancellable"}

            # Mark as cancelled
            task.status = "cancelled"
            task.error = "Conversion cancelled by user"

            # Clean up temporary files if they exist (only for processing tasks)
            if task.status_was == "processing" and task.output_file:
                temp_file = f"{task.output_file}.converting"
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.info(f"Removed temporary file for cancelled task {task_id}: {temp_file}")
                    except Exception as e:
                        logger.error(f"Failed to remove temporary file for task {task_id}: {e}")

            if task_id in active_tasks:
                active_tasks.remove(task_id)
                logger.info(f"Marked pending task {task_id} as cancelled for scene: {task.scene.title}")
            else:
                logger.info(f"Task {task_id} has status {task.status}, not cancelling")
                return {"status": "not_cancellable"}

            try:
                with open(task.log_file, 'a') as log:
                    log.write(f"\n--- Conversion cancelled by user ---\n")
                    log.write(f"Task cancelled at: {time.time()}\n")
                    log.write(f"Status changed from {task.status} to cancelled\n")
            except Exception as e:
                logger.error(f"Failed to write cancellation to log: {e}")

            save_queue_state()
            break

    return {"status": "cancelled"}

@app.post("/api/clear-completed")
async def clear_completed():
    global conversion_queue
    tasks_to_keep = [task for task in conversion_queue if task.status in ["pending", "processing", "cancelled"]]
    tasks_removed = len(conversion_queue) - len(tasks_to_keep)

    # Clean up temporary files for cancelled tasks being removed
    for task in conversion_queue:
        if task.status == "cancelled" and task not in tasks_to_keep:
            if task.output_file and os.path.exists(task.output_file + ".converting"):
                try:
                    os.remove(task.output_file + ".converting")
                    logger.info(f"Cleaned up temporary file for removed cancelled task {task.task_id}")
                except Exception as e:
                    logger.error(f"Failed to clean up temporary file for task {task.task_id}: {e}")

    conversion_queue = tasks_to_keep
    save_queue_state()
    logger.info(f"Cleared {tasks_removed} completed/error tasks from queue")
    return {"status": "cleared"}

@app.post("/api/cancel-all-conversions")
async def cancel_all_conversions():
    global conversion_queue
    cancelled_count = 0

    # Only cancel tasks that can be cancelled (processing or pending)
    task_ids = [task.task_id for task in conversion_queue if task.status in ["processing", "pending"]]

    for task_id in task_ids:
        try:
            await cancel_conversion(task_id)
            cancelled_count += 1
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")

    save_queue_state()

    return {"status": "cancelled", "count": cancelled_count}

@app.post("/api/toggle-pause")
async def toggle_pause():
    global queue_paused
    queue_paused = not queue_paused
    clear_sse_cache()

    if not queue_paused and conversion_queue:
        pending_tasks = [task for task in conversion_queue if task.status == "pending"]
        if pending_tasks:
            asyncio.create_task(process_conversion_queue())

    return {"status": "ok", "paused": queue_paused}

@app.post("/api/start-processing")
async def start_processing():
    clear_sse_cache()
    return {"status": "processing_started"}

@app.post("/api/remove-from-queue/{task_id}")
async def remove_from_queue(task_id: str):
    """Remove a task from the queue, cleaning up temporary files if cancelled"""
    global conversion_queue

    try:
        # Find the task to remove
        task_index = next((i for i, t in enumerate(conversion_queue) if t.task_id == task_id), -1)
        if task_index == -1:
            raise HTTPException(status_code=404, detail="Task not found")

        task = conversion_queue[task_index]

        # Clean up temporary files if task was processing or cancelled
        temp_file = None
        if task.output_file:
            temp_file = f"{task.output_file}.converting"

        # Also check for .converting suffix in output_file itself
        if task.output_file and task.output_file.endswith('.converting'):
            temp_file = task.output_file

        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file for removed task {task_id}: {temp_file}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary file for task {task_id}: {e}")

        # Remove from queue
        conversion_queue = [task for task in conversion_queue if str(task.task_id) != str(task_id)]

        save_queue_state()
        return {"status": "removed", "task_id": task_id, "status_was": task.status}

    except Exception as e:
        logger.error(f"Failed to remove task {task_id} from queue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove task: {str(e)}")

@app.post("/api/remove-all-pending")
async def remove_all_pending():
    global conversion_queue

    try:
        pending_count = len([task for task in conversion_queue if task.status == 'pending'])
        conversion_queue = [task for task in conversion_queue if task.status != 'pending']

        save_queue_state()
        logger.info(f"Removed {pending_count} pending tasks from queue")
        return {"status": "removed", "count": pending_count}
    except Exception as e:
        logger.error(f"Failed to remove pending tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove pending tasks: {str(e)}")

@app.post("/api/retry-conversion/{task_id}")
async def retry_conversion(task_id: str):
    """Retry a failed conversion task"""
    global conversion_queue

    try:
        task_index = next((i for i, t in enumerate(conversion_queue) if t.task_id == task_id), -1)
        if task_index == -1:
            raise HTTPException(status_code=404, detail="Task not found")

        task = conversion_queue[task_index]

        # Allow retrying error and cancelled tasks
        if task.status not in ["error", "cancelled"]:
            raise HTTPException(status_code=400, detail=f"Cannot retry task with status: {task.status}")

        # Reset task status
        task.status = "pending"
        task.progress = 0.0
        task.eta = None
        task.error = None

        save_queue_state()
        clear_sse_cache()

        logger.info(f"Retrying conversion task {task_id} for scene: {task.scene.title}")

        if not queue_paused:
            asyncio.create_task(process_conversion_queue())

        return {"status": "retried"}
    except Exception as e:
        logger.error(f"Failed to retry conversion task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry conversion: {str(e)}")

@app.post("/api/retry-stash-fix/{task_id}")
async def retry_stash_fix(task_id: str):
    """Specifically retry only the Stash update for a completed_with_warning task"""
    global conversion_queue

    try:
        task_index = next((i for i, t in enumerate(conversion_queue) if t.task_id == task_id), -1)
        if task_index == -1:
            raise HTTPException(status_code=404, detail="Task not found")

        task = conversion_queue[task_index]

        if task.status != "completed_with_warning":
            raise HTTPException(status_code=400, detail=f"Cannot fix Stash for task with status: {task.status}")

        # Check if output file exists
        if not task.output_file or not os.path.exists(task.output_file):
            # Output file is missing. This could mean:
            # 1. User deleted the file
            # 2. File was renamed/moved
            # 3. File was manually imported into Stash
            # Reset task to pending so user can retry the full conversion
            task.status = "pending"
            task.progress = 0.0
            task.eta = None
            task.error = "Output file missing. Reset to pending for full retry."
            save_queue_state()
            clear_sse_cache()
            logger.warning(f"[Stash Fix] Output file missing for task {task_id}, reset to pending")
            raise HTTPException(
                status_code=400,
                detail="Output file not found. Task has been reset to pending. You can retry the full conversion or remove the task."
            )

        logger.info(f"[Fix Stash] Retrying Stash update for task {task_id}, output file: {task.output_file}")

        # Create a special task that will only do Stash update
        asyncio.create_task(fix_stash_update(task))

        return {"status": "retrying_stash"}
    except HTTPException:
        # Re-raise HTTP exceptions so they reach the client with proper details
        raise
    except Exception as e:
        logger.error(f"Failed to start Stash fix for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start Stash fix: {str(e)}")

async def find_available_filename(base_name: str, container: str, max_attempts: int = 100) -> str:
    """Find an available filename, checking for existing converted files"""
    # First, check if the base name already exists with this extension
    if not os.path.exists(f"{base_name}.{container}"):
        return f"{base_name}.{container}"

    # Check for existing converted files with pattern base_name_N.container
    # where N is a number
    import re
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.{re.escape(container)}$")

    existing_numbers = []
    base_dir = os.path.dirname(base_name)
    base_file = os.path.basename(base_name)

    if os.path.exists(base_dir):
        for filename in os.listdir(base_dir):
            match = pattern.match(filename)
            if match:
                existing_numbers.append(int(match.group(1)))

    if existing_numbers:
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1

    # Make sure we don't exceed max attempts
    for i in range(next_number, next_number + max_attempts):
        candidate = f"{base_name}_{i}.{container}"
        if not os.path.exists(candidate):
            return candidate

    # Fallback: use timestamp
    import time
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}.{container}"

async def trigger_stash_scan(file_path: str):
    config = get_config()
    path_mappings = config.get('path_mappings', [])

    docker_path = apply_path_mappings(file_path, path_mappings)

    scan_mutation = """
    mutation ScanFile($path: String!) {
      metadataScan(input: { paths: [$path] })
    }
    """
    try:
        await stash_request(scan_mutation, {"path": docker_path})
        logger.info(f"Triggered metadata scan for: {docker_path}")
    except Exception as e:
        logger.warning(f"Metadata scan failed (non-critical): {e}")
        raise Exception(f"Stash scan failed: {str(e)}")

async def add_file_to_scene(scene_id: str, new_file_path: str, overwrite_original: bool):
    """Add a new file to a scene in Stash (for non-overwrite mode)"""
    config = get_config()
    path_mappings = config.get('path_mappings', [])

    # Convert the host path back to Docker path for Stash
    docker_path = apply_path_mappings(new_file_path, path_mappings)
    new_basename = os.path.basename(docker_path)

    logger.info(f"[Add File] Attempting to add file {new_basename} to scene {scene_id}")

    # First, let's see if the file is already in the scene
    # This avoids unnecessary scans and queries
    check_scene_files_query = """
    query FindSceneWithFiles($scene_id: ID!) {
      findScene(id: $scene_id) {
        files {
          id
          path
          basename
        }
      }
    }
    """

    try:
        scene_result = await stash_request(check_scene_files_query, {"scene_id": scene_id})
        if scene_result['data']['findScene']['files']:
            for file in scene_result['data']['findScene']['files']:
                if file['basename'] == new_basename or file['path'] == docker_path:
                    logger.info(f"[Add File] File {new_basename} is already in scene {scene_id}")
                    return  # Already in scene
    except Exception as e:
        logger.warning(f"[Add File] Could not check scene files: {e}")
        # Continue anyway

    # Try to scan the file first (but don't fail if it doesn't work)
    try:
        await trigger_stash_scan(new_file_path)
        logger.info(f"[Add File] File scan triggered")
    except Exception as scan_error:
        logger.warning(f"[Add File] Scan might have failed: {scan_error}")
        # Continue anyway - file might already be scanned

    # Wait a bit for scan to complete
    await asyncio.sleep(2)

    # Try to find the scene id of the new file path
    find_scene_of_file_query = """
    query FindSceneByPath($scene_filter: SceneFilterType!){
      findScenes(scene_filter: $scene_filter){
        scenes{
          id
        }
      }
    }
    """

    scene_of_file_id = None
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            logger.info(f"[Add File] Trying to find new scene of converted file")
            scene_of_file_result = await stash_request_with_retry(find_scene_of_file_query, {"scene_filter": {"path": {"value": docker_path, "modifier": "EQUALS"}}})
            if scene_of_file_result['data']['findScenes']['scenes']:
                scene_of_file_id = scene_of_file_result['data']['findScenes']['scenes'][0]['id']
                logger.info(f"[Add File] Found scene of file in Stash with ID: {scene_of_file_id}")
                break
            else:
                if attempt < max_attempts - 1:
                    logger.info(f"[Add File] Scene of File not found, waiting 3 seconds (attempt {attempt + 1}/{max_attempts})...")
                    await asyncio.sleep(3)
                else:
                    raise Exception(f"File not found in Stash: {docker_path}")
        except HTTPException as e:
            if attempt < max_attempts - 1:
                logger.warning(f"[Add File] Query failed, retrying... (attempt {attempt + 1}/{max_attempts}): {e.detail}")
                await asyncio.sleep(3)
            else:
                # If we can't find the file, maybe we can add it by path directly
                # Some Stash versions support adding by path without file_id
                logger.warning(f"[Add File] Could not find file ID, trying alternative method...")
                break
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.warning(f"[Add File] Query failed, retrying... (attempt {attempt + 1}/{max_attempts}): {e}")
                await asyncio.sleep(3)
            else:
                raise Exception(f"Failed to find file in Stash: {str(e)}")

    if scene_of_file_id:
        # Merging the scene with the new file into the existing scene
        merge_mutation = f"""
        mutation SceneMerge{{
          sceneMerge(
            input: {{
              source: {scene_of_file_id},
              destination: {scene_id}
            }}
          ) {{
            id
          }}
        }}
        """

        try:
            result = await stash_request(merge_mutation)
            logger.info(f"[Add File] Successfully added file {new_basename} to scene {scene_id}")
            return
        except HTTPException as e:
            error_detail = str(e.detail).lower()
            if "already assigned" in error_detail or "already exists" in error_detail or "duplicate" in error_detail:
                logger.info(f"[Add File] File was already assigned to scene")
                return
            else:
                logger.warning(f"[Add File] Assignment failed: {e.detail}")
        except Exception as e:
            logger.warning(f"[Add File] Assignment failed: {e}")

    # Final fallback: log that manual intervention might be needed
    logger.error(f"[Add File] All methods failed to add file {new_basename} to scene {scene_id}")
    logger.error(f"[Add File] File exists at: {docker_path}")
    logger.error(f"[Add File] Manual steps might be needed in Stash web interface")

    raise Exception(f"All methods failed to add file to scene. File exists at {docker_path}")

async def stash_request_with_retry(graphql_query: str, variables: dict = None, max_retries: int = 3):
    """Make a Stash request with retry logic"""
    for attempt in range(max_retries):
        try:
            return await stash_request(graphql_query, variables)
        except HTTPException as e:
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)  # Exponential backoff: 2, 4, 6 seconds
                logger.warning(f"Stash request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e.detail}")
                logger.warning(f"Stash query:\n{graphql_query}\nVariables:\n{variables}")
                await asyncio.sleep(wait_time)
            else:
                raise

async def update_stash_file(scene_id: str, file_id: str, new_file_path: str, overwrite_original: bool):
    config = get_config()
    path_mappings = config.get('path_mappings', [])

    docker_path = apply_path_mappings(new_file_path, path_mappings)
    new_basename = os.path.basename(docker_path)

    import time
    updated_at = int(time.time())

    sql = f"UPDATE files SET basename = '{new_basename}', updated_at = {updated_at} WHERE id = {file_id};"

    exec_sql_mutation = f"""
    mutation ExecSQL {{
      execSQL( sql: "{sql}" ) {{
        rows_affected
      }}
     }}
     """

    try:
        logger.info(f"Updating Stash file {file_id} with execSQL: {sql}")
        await stash_request(exec_sql_mutation, {})
        logger.info(f"Successfully updated Stash file {file_id} with new basename: {new_basename}")
    except Exception as e:
        logger.error(f"Failed to update Stash file via execSQL: {e}")
        raise Exception(f"Stash database update failed: {str(e)}")

    if overwrite_original:
        scan_mutation = """
        mutation ScanFile($path: String!) {
          metadataScan(input: { paths: [$path] })
        }
        """
        try:
            await stash_request(scan_mutation, {"path": docker_path})
            logger.info(f"Triggered metadata scan for: {docker_path}")
        except Exception as e:
            logger.warning(f"Metadata scan failed (non-critical): {e}")

@app.get("/sse")
async def sse_endpoint(request: Request):
    async def event_generator():
        global last_sse_data
        client_id = str(uuid.uuid4())
        sse_clients.add(client_id)

        try:
            while True:
                if await request.is_disconnected():
                    break

                serializable_queue = []
                for task in conversion_queue:
                    task_data = {
                        "task_id": task.task_id,
                        "scene": {
                            "id": task.scene.id,
                            "title": task.scene.title,
                            "files": [file.dict() for file in task.scene.files] if task.scene.files else []
                        },
                        "status": task.status,
                        "progress": task.progress,
                        "eta": task.eta,
                        "output_file": task.output_file,
                        "error": task.error
                    }
                    serializable_queue.append(task_data)

                current_config = get_config()
                global queue_paused
                status_data = {
                    "queue": serializable_queue,
                    "active": list(active_tasks),
                    "paused": queue_paused
                }

                if last_sse_data != status_data:
                    last_sse_data = status_data
                    yield f"data: {json.dumps(status_data)}\n\n"
                else:
                    await asyncio.sleep(1)
                    continue
                await asyncio.sleep(1)
        finally:
            sse_clients.discard(client_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

def convert_bitrate_to_bps(bitrate_str: str) -> int:
    multipliers = {'k': 1000, 'm': 1000000, 'g': 1000000000}
    if bitrate_str and bitrate_str[-1].lower() in multipliers:
        return int(bitrate_str[:-1]) * multipliers[bitrate_str[-1].lower()]
    return int(bitrate_str) if bitrate_str else 0

if __name__ == "__main__":
    uvicorn.run("stash_shrink:app", host="0.0.0.0", port=9899, reload=True)
