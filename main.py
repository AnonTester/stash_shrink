import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid
import subprocess
import shutil
import re
import time

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stash Shrink")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
CONFIG_FILE = "config.json"
LOGS_DIR = "logs"

# Default configuration
DEFAULT_CONFIG = {
    "stash_url": "http://localhost:9999",
    "api_key": "",
    "default_search_limit": 50,
    "max_concurrent_tasks": 2,
    "video_settings": {
        "width": 1280,
        "height": 720,
        "bitrate": "1000k",
        "framerate": 30,
        "buffer_size": "2000k",
        "container": "mp4"
    }
}

# Global state
conversion_queue = []
conversion_tasks = {}
active_tasks = set()
task_status = {}
sse_clients = set()

# Ensure logs directory exists
Path(LOGS_DIR).mkdir(exist_ok=True)

class Settings(BaseModel):
    stash_url: str
    api_key: str
    default_search_limit: int
    max_concurrent_tasks: int
    video_settings: Dict[str, Any]

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

    # For backward compatibility with existing code that expects path and file
    @property
    def path(self):
        """Return the path of the first file"""
        return self.files[0].path if self.files else ""

    @property
    def file(self):
        """Return the first file's data as a dict for backward compatibility"""
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
    status: str = "pending"  # pending, processing, completed, error
    progress: float = 0.0
    eta: Optional[float] = None
    log_file: str
    output_file: Optional[str] = None
    error: Optional[str] = None

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
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
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as e:
        logger.error(f"Connection error to Stash at {config['stash_url']}: {e}")
        logger.error(f"GraphQL Query that failed: {graphql_query}")
        if variables:
            logger.error(f"GraphQL Variables: {variables}")
        raise HTTPException(
            status_code=400,
            detail=f"Cannot connect to Stash at {config['stash_url']}. Please check if the URL is correct and Stash is running."
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to Stash: {e}")
        logger.error(f"GraphQL Query that failed: {graphql_query}")
        if variables:
            logger.error(f"GraphQL Variables: {variables}")
        raise HTTPException(
            status_code=400,
            detail=f"Connection timeout to Stash at {config['stash_url']}. Please check if Stash is running and accessible."
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Stash: {e.response.status_code}")
        logger.error(f"GraphQL Query that failed: {graphql_query}")
        if variables:
            logger.error(f"GraphQL Variables: {variables}")

        # Try to get more details from the response
        error_detail = f"Stash returned error {e.response.status_code}"
        try:
            error_data = e.response.json()
            if 'errors' in error_data:
                error_messages = [err.get('message', 'Unknown error') for err in error_data['errors']]
                error_detail += f": {', '.join(error_messages)}"
        except:
            pass

        raise HTTPException(
            status_code=400,
            detail=error_detail
        )
    except Exception as e:
        logger.error(f"Unexpected error during Stash request: {e}")
        logger.error(f"GraphQL Query that failed: {graphql_query}")
        if variables:
            logger.error(f"GraphQL Variables: {variables}")
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected error contacting Stash: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    config = get_config()
    show_settings = is_first_run()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": config,
        "show_settings": show_settings
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
        # Use the provided GraphQL query structure
        query = """
        query FindAllScenes {
          findScenes {
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
            logger.error(f"GraphQL errors in response: {result['errors']}")
            raise HTTPException(status_code=400, detail=f"GraphQL error: {result['errors']}")

        if 'data' not in result or 'findScenes' not in result['data']:
            logger.error(f"Unexpected response structure: {result}")
            raise HTTPException(status_code=400, detail="Unexpected response structure from Stash")

        scenes_data = result['data']['findScenes']['scenes']
        logger.info(f"Found {len(scenes_data)} scenes in Stash")

        scenes = []
        for scene_data in scenes_data:
            try:
                # Only include scenes that have files
                if scene_data.get('files'):
                    # Convert file data to SceneFile objects
                    files = []
                    for file_data in scene_data['files']:
                        try:
                            # Handle potential None values by providing defaults
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
                            scene_file = SceneFile(**processed_file_data)
                            files.append(scene_file)
                        except Exception as e:
                            logger.warning(f"Failed to parse file data: {file_data}, error: {e}")
                            continue

                    # Apply filters - a file should be included if it exceeds ANY of the max limits
                    # OR doesn't match the specified codec
                    filtered_files = []
                    for file in files:
                        include_file = False

                        # Check if file exceeds any of the maximum limits
                        # If no limits are specified for a field, we don't filter by it
                        exceeds_limits = False

                        # Check width
                        if search_params.max_width and file.width and file.width > search_params.max_width:
                            exceeds_limits = True
                            logger.debug(f"File {file.basename} exceeds width: {file.width} > {search_params.max_width}")

                        # Check height
                        if search_params.max_height and file.height and file.height > search_params.max_height:
                            exceeds_limits = True
                            logger.debug(f"File {file.basename} exceeds height: {file.height} > {search_params.max_height}")

                        # Check bitrate
                        if search_params.max_bitrate and file.bit_rate:
                            bitrate_value = convert_bitrate_to_bps(search_params.max_bitrate)
                            if file.bit_rate > bitrate_value:
                                exceeds_limits = True
                                logger.debug(f"File {file.basename} exceeds bitrate: {file.bit_rate} > {bitrate_value}")

                        # Check framerate
                        if search_params.max_framerate and file.frame_rate and file.frame_rate > search_params.max_framerate:
                            exceeds_limits = True
                            logger.debug(f"File {file.basename} exceeds framerate: {file.frame_rate} > {search_params.max_framerate}")

                        # Check codec - include if codec doesn't match (we want to convert files with wrong codec)
                        wrong_codec = False
                        if search_params.codec and file.video_codec:
                            # Normalize codec names for comparison
                            file_codec = file.video_codec.lower().replace('.', '')
                            search_codec = search_params.codec.lower().replace('.', '')
                            if file_codec != search_codec:
                                wrong_codec = True
                                logger.debug(f"File {file.basename} has wrong codec: {file.video_codec} != {search_params.codec}")

                        # Check path filter
                        path_matches = True
                        if search_params.path and search_params.path not in file.path:
                            path_matches = False
                            logger.debug(f"File {file.basename} doesn't match path filter: {file.path}")

                        # Include file if it exceeds any limits OR has wrong codec, AND matches path filter
                        if (exceeds_limits or wrong_codec) and path_matches:
                            include_file = True

                        # If no filters are specified, include all files
                        if not any([search_params.max_width, search_params.max_height,
                                   search_params.max_bitrate, search_params.max_framerate,
                                   search_params.codec, search_params.path]):
                            include_file = True

                        if include_file:
                            filtered_files.append(file)

                    # Only include scene if it has files after filtering
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
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search_scenes: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during search: {str(e)}"
        )

@app.post("/api/queue-conversion")
async def queue_conversion(scene_ids: List[str]):
    config = get_config()

    try:
        # Get scene details from Stash using the new query structure
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
            except Exception as e:
                logger.error(f"Failed to queue scene {scene_data.get('id', 'unknown')}: {e}")
                continue

        # Start processing if not already running
        if queued_count > 0 and len(active_tasks) < config['max_concurrent_tasks']:
            asyncio.create_task(process_conversion_queue())

        return {"status": "queued", "count": queued_count}

    except Exception as e:
        logger.error(f"Error queueing conversion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to queue conversion: {str(e)}")

# ... [Keep the rest of the existing functions like process_conversion_queue, convert_video, etc.] ...

class FFmpegProgress:
    def __init__(self, total_duration: float):
        self.total_duration = total_duration
        self.current_time = 0.0
        self.progress = 0.0

    def parse_line(self, line: str):
        """Parse FFmpeg output line to extract progress information"""
        line = line.strip()

        # Try to parse time from various FFmpeg output formats
        time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
        if time_match:
            time_str = time_match.group(1)
            self.current_time = self.parse_time_string(time_str)
            if self.total_duration > 0:
                self.progress = min((self.current_time / self.total_duration) * 100, 100.0)
            return True

        # Alternative time format
        time_match = re.search(r'time=(\d+)', line)
        if time_match:
            self.current_time = float(time_match.group(1))
            if self.total_duration > 0:
                self.progress = min((self.current_time / self.total_duration) * 100, 100.0)
            return True

        return False

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

async def convert_video(task: ConversionTask):
    config = get_config()
    video_settings = config['video_settings']

    try:
        # Use the first file from the scene
        if not task.scene.files or len(task.scene.files) == 0:
            raise Exception("No files found in scene")

        scene_file = task.scene.files[0]
        input_file = scene_file.path
        base_name = os.path.splitext(input_file)[0]
        temp_output = f"{base_name}.converting.{video_settings['container']}"
        final_output = await find_available_filename(base_name, video_settings['container'])

        # Get file duration for progress calculation
        file_duration = scene_file.duration or 0
        progress_tracker = FFmpegProgress(file_duration)

        # Build FFmpeg command with progress output
        ffmpeg_cmd = build_ffmpeg_command(input_file, temp_output, video_settings)

        # Write to log
        with open(task.log_file, 'a') as log:
            log.write(f"Starting conversion: {input_file} -> {final_output}\n")
            log.write(f"File duration: {file_duration} seconds\n")
            log.write(f"FFmpeg command: {ffmpeg_cmd}\n")
            log.write("-" * 80 + "\n")

        # Start time for ETA calculation
        start_time = time.time()

        # Run FFmpeg and capture progress
        process = await asyncio.create_subprocess_shell(
            ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            universal_newlines=True
        )

        # Read stderr line by line to capture progress
        while True:
            line = await process.stderr.readline()
            if not line:
                break

            line = line.strip()

            # Write to log
            with open(task.log_file, 'a') as log:
                log.write(line + '\n')

            # Parse progress
            if progress_tracker.parse_line(line):
                task.progress = progress_tracker.progress

                # Calculate ETA
                if task.progress > 0:
                    elapsed_time = time.time() - start_time
                    total_estimated_time = elapsed_time / (task.progress / 100)
                    remaining_time = total_estimated_time - elapsed_time
                    task.eta = remaining_time

            # Check if process has finished
            if process.returncode is not None:
                break

        # Wait for process to complete
        stdout, stderr = await process.communicate()

        # Final log entry
        with open(task.log_file, 'a') as log:
            log.write("-" * 80 + "\n")
            log.write(f"FFmpeg process completed with return code: {process.returncode}\n")
            if stdout:
                log.write(f"STDOUT: {stdout}\n")
            if stderr:
                log.write(f"STDERR: {stderr}\n")

        if process.returncode == 0:
            # Verify the output file was created
            if not os.path.exists(temp_output):
                raise Exception("Output file was not created")

            # Get file size for verification
            output_size = os.path.getsize(temp_output)
            if output_size == 0:
                raise Exception("Output file is empty")

            # Rename temporary file to final name
            os.rename(temp_output, final_output)

            # Update Stash with the new file
            await update_stash_file(task.scene.id, scene_file.id, final_output)

            # Delete original file
            if os.path.exists(input_file):
                os.remove(input_file)

            task.status = "completed"
            task.output_file = final_output
            task.progress = 100.0

            logger.info(f"Successfully converted {input_file} to {final_output}")
        else:
            task.status = "error"
            task.error = f"FFmpeg failed with return code {process.returncode}"
            logger.error(f"FFmpeg conversion failed for {input_file}")

    except Exception as e:
        task.status = "error"
        task.error = str(e)
        logger.error(f"Conversion failed for {task.scene.files[0].path if task.scene.files else 'unknown'}: {e}")

        # Clean up temporary file if it exists
        if 'temp_output' in locals() and os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass

def build_ffmpeg_command(input_file: str, output_file: str, settings: Dict[str, Any]) -> str:
    framerate_option = f"-r {settings['framerate']}" if settings.get('framerate') else ""

    cmd = f"""ffmpeg -i "{input_file}" -filter_complex "scale=ceil(iw*min(1,min({settings['width']}/iw,{settings['height']}/ih))/2)*2:-2" -c:v libx264 {framerate_option} -crf 28 -c:a aac -b:v {settings['bitrate']} -maxrate {settings['bitrate']} -buffersize {settings['buffer_size']} -f {settings['container']} "{output_file}" """

    return cmd

async def find_available_filename(base_name: str, container: str) -> str:
    counter = 1
    while True:
        if counter == 1:
            candidate = f"{base_name}.{container}"
        else:
            candidate = f"{base_name}_{counter}.{container}"

        if not os.path.exists(candidate):
            return candidate
        counter += 1

async def update_stash_file(scene_id: str, file_id: str, new_file_path: str):
    """Update Stash with the new file information"""
    config = get_config()

    # This mutation would update the file path in Stash
    # You'll need to adjust this based on your Stash GraphQL schema
    mutation = """
    mutation UpdateFile($id: ID!, $path: String!) {
      fileUpdate(input: { id: $id, path: $path }) {
        id
        path
      }
    }
    """

    variables = {
        "id": file_id,
        "path": new_file_path
    }

    try:
        await stash_request(mutation, variables)
        logger.info(f"Updated Stash file {file_id} with new path: {new_file_path}")
    except Exception as e:
        logger.error(f"Failed to update Stash file {file_id}: {e}")
        # Don't fail the conversion if Stash update fails

def convert_bitrate_to_bps(bitrate_str: str) -> int:
    """Convert bitrate string (e.g., '1000k') to bits per second"""
    multipliers = {'k': 1000, 'm': 1000000, 'g': 1000000000}
    if bitrate_str and bitrate_str[-1].lower() in multipliers:
        return int(bitrate_str[:-1]) * multipliers[bitrate_str[-1].lower()]
    return int(bitrate_str) if bitrate_str else 0

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9899, reload=True)
