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

# Configuration
CONFIG_FILE = "config.json"
LOGS_DIR = "logs"

# Default configuration
DEFAULT_CONFIG = {
    "stash_url": "http://localhost:9999",
    "api_key": "",
    "overwrite_original": True,  # Default to overwrite behavior
    "default_search_limit": 50,
    "max_concurrent_tasks": 2,
    "video_settings": {
        "width": 1280,
        "height": 720,
        "bitrate": "1000k",
        "framerate": 30,
        "buffer_size": "2000k",
        "container": "mp4"
    },
    "path_mappings": []
}

# Global state
queue_initialized = False
last_sse_data = None  # Cache last sent SSE data to prevent duplicate sends
conversion_queue = []
conversion_tasks = {}
active_tasks = set()
QUEUE_STATE_FILE = "conversion_queue.json"
task_status = {}
sse_clients = set()

# Ensure logs directory exists
Path(LOGS_DIR).mkdir(exist_ok=True)

class Settings(BaseModel):
    stash_url: str
    api_key: str
    default_search_limit: int
    max_concurrent_tasks: int
    overwrite_original: bool = True  # New setting for overwrite behavior
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

# Lifespan event handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Stash Shrink")
    initialize_queue_system()
    # Queue starts paused by default
    global queue_paused
    queue_paused = True
    yield
    # Shutdown
    logger.info("Shutting down Stash Shrink")
    # Clean up any active tasks
    for task_id in list(active_tasks):
        task = task_status.get(task_id)
        if task and task.status == "processing":
            task.status = "pending"  # Reset to pending so it can be resumed
            logger.info(f"Reset active task {task_id} to pending on shutdown")
    save_queue_state()  # Save queue state on shutdown

# Create FastAPI app with lifespan
app = FastAPI(title="Stash Shrink", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Migrate from delete_original to overwrite_original if needed
            if 'delete_original' in config and 'overwrite_original' not in config:
                config['overwrite_original'] = config['delete_original']
                # Remove old setting
                del config['delete_original']
                # Save migrated config
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
            # Convert task to serializable format
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

    # Clear existing queue to avoid duplicates
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

            # Ensure paths end with slash for proper replacement
            docker_path_norm = docker_path.rstrip('/') + '/'
            host_path_norm = host_path.rstrip('/') + '/'

            # Replace Docker path with host path
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

            # Log the full response for debugging
            logger.debug(f"Stash response status: {response.status_code}")
            if response.status_code != 200:
                logger.debug(f"Stash response body: {response.text}")

            response.raise_for_status()
            result = response.json()

            # Log the structure of the response for debugging
            if 'data' in result and 'findScenes' in result['data']:
                scenes_count = len(result['data']['findScenes'].get('scenes', []))
                reported_count = result['data']['findScenes'].get('count', 0)
                logger.debug(f"Stash response: {scenes_count} scenes returned, {reported_count} reported total")

            return result
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

        # Log the response body for more details
        try:
            error_body = e.response.text
            logger.error(f"Stash response body: {error_body}")
        except:
            pass

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
        # Use the provided GraphQL query structure with explicit pagination
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
            logger.error(f"GraphQL errors in response: {result['errors']}")
            # If per_page: -1 doesn't work, try a large number
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
                logger.error(f"GraphQL errors in fallback response: {result['errors']}")
                raise HTTPException(status_code=400, detail=f"GraphQL error: {result['errors']}")

        if 'data' not in result or 'findScenes' not in result['data']:
            logger.error(f"Unexpected response structure: {result}")
            raise HTTPException(status_code=400, detail="Unexpected response structure from Stash")

        scenes_data = result['data']['findScenes']['scenes']
        actual_count = len(scenes_data)
        reported_count = result['data']['findScenes'].get('count', actual_count)

        logger.info(f"Found {actual_count} scenes in Stash (reported count: {reported_count})")

        # If we're getting fewer scenes than expected, log a warning
        if reported_count > actual_count:
            logger.warning(f"Stash returned {actual_count} scenes but reported {reported_count}. There might be a pagination limit.")

        scenes = []
        config = get_config()
        path_mappings = config.get('path_mappings', [])

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

                            # Apply path mappings to file path
                            if processed_file_data['path']:
                                processed_file_data['path'] = apply_path_mappings(
                                    processed_file_data['path'], path_mappings
                                )

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
                        exceeds_limits = False

                        # Check width (only if file has width and we're filtering by width)
                        if search_params.max_width is not None and file.width is not None:
                            if file.width > search_params.max_width:
                                exceeds_limits = True
                                logger.debug(f"File {file.basename} exceeds width: {file.width} > {search_params.max_width}")

                        # Check height
                        if search_params.max_height is not None and file.height is not None:
                            if file.height > search_params.max_height:
                                exceeds_limits = True
                                logger.debug(f"File {file.basename} exceeds height: {file.height} > {search_params.max_height}")

                        # Check bitrate
                        if search_params.max_bitrate and file.bit_rate:
                            bitrate_value = convert_bitrate_to_bps(search_params.max_bitrate)
                            if file.bit_rate > bitrate_value:
                                exceeds_limits = True
                                logger.debug(f"File {file.basename} exceeds bitrate: {file.bit_rate} > {bitrate_value}")

                        # Check framerate
                        if search_params.max_framerate is not None and file.frame_rate is not None:
                            if file.frame_rate > search_params.max_framerate:
                                exceeds_limits = True
                                logger.debug(f"File {file.basename} exceeds framerate: {file.frame_rate} > {search_params.max_framerate}")

                        # Check codec - include if codec doesn't match (we want to convert files with wrong codec)
                        wrong_codec = False
                        if search_params.codec and file.video_codec:
                            # Normalize codec names for comparison - FIXED PYTHON SYNTAX
                            file_codec = (file.video_codec or '').lower().replace('.', '')
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
                        if not any([
                            search_params.max_width is not None,
                            search_params.max_height is not None,
                            search_params.max_bitrate is not None,
                            search_params.max_framerate is not None,
                            search_params.codec is not None,
                            search_params.path is not None
                        ]):
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

@app.get("/api/conversion-log/{task_id}")
async def get_conversion_log(task_id: str):
    """Get the log content for a specific conversion task"""
    try:
        # Find the task in the queue
        task = next((t for t in conversion_queue if t.task_id == task_id), None)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Read the log file
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

    # Check for duplicates
    existing_scene_ids = {task.scene.id for task in conversion_queue}
    new_scene_ids = set(scene_ids) - existing_scene_ids
    scene_ids = list(new_scene_ids)

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
                    # Apply path mappings to file paths
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
                    save_queue_state()  # Save after adding to queue
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

@app.get("/api/conversion-status")
async def conversion_status():
    global queue_paused
    config = get_config()

    return {
        "queue": [task.dict() for task in conversion_queue],
        "active": list(active_tasks),
        "completed": [task.dict() for task in conversion_queue if task.status in ["completed", "error"]],
        "paused": config.get('paused', True)  # Default to paused
    }

@app.post("/api/cancel-conversion/{task_id}")
async def cancel_conversion(task_id: str):
    for i, task in enumerate(conversion_queue):
        if task.task_id == task_id:
            # Mark task as cancelled
            task.status = "cancelled"
            task.error = "Conversion cancelled by user"

            # Remove from active tasks
            if task_id in active_tasks:
                active_tasks.remove(task_id)

            # Clean up temporary files if they exist
            if task.output_file and os.path.exists(task.output_file):
                try:
                    os.remove(task.output_file)
                    logger.info(f"Cleaned up output file for cancelled task: {task.output_file}")
                except Exception as e:
                    logger.error(f"Failed to clean up output file {task.output_file}: {e}")

            # Write cancellation to log
            try:
                with open(task.log_file, 'a') as log:
                    log.write(f"\n--- Conversion cancelled by user ---\n")
            except Exception as e:
                logger.error(f"Failed to write cancellation to log: {e}")

            logger.info(f"Cancelled conversion task {task_id} for scene: {task.scene.title}")

            # Remove from queue
            conversion_queue.pop(i)
            task_status.pop(task_id, None)
            break

    save_queue_state()  # Save after cancellation
    return {"status": "cancelled"}

@app.post("/api/clear-completed")
async def clear_completed():
    global conversion_queue
    tasks_to_keep = [task for task in conversion_queue if task.status in ["pending", "processing"]]
    tasks_removed = len(conversion_queue) - len(tasks_to_keep)
    conversion_queue = tasks_to_keep
    save_queue_state()
    logger.info(f"Cleared {tasks_removed} completed/error tasks from queue")
    return {"status": "cleared"}

@app.post("/api/cancel-all-conversions")
async def cancel_all_conversions():
    global conversion_queue
    cancelled_count = 0

    # Create a copy of task IDs to avoid modification during iteration
    task_ids = [task.task_id for task in conversion_queue]

    for task_id in task_ids:
        try:
            await cancel_conversion(task_id)
            cancelled_count += 1
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")

    # Clear the queue and save state
    conversion_queue = []
    save_queue_state()

    return {"status": "cancelled", "count": cancelled_count}

# Runtime queue state (not persisted)
# Initialize as paused by default
def initialize_queue_state():
    global queue_paused
    queue_paused = True

@app.post("/api/toggle-pause")
async def toggle_pause():
    global queue_paused
    queue_paused = not queue_paused
    clear_sse_cache()  # Force SSE update when pause state changes
    return {"status": "ok", "paused": queue_paused}

@app.post("/api/start-processing")
async def start_processing():
    """Start processing the queue if not paused"""
    global queue_paused
    config = get_config()
    if not queue_paused and conversion_queue and len(active_tasks) < config['max_concurrent_tasks']:
        asyncio.create_task(process_conversion_queue())
    clear_sse_cache()  # Force SSE update
    return {"status": "processing_started"}

@app.post("/api/remove-from-queue/{task_id}")
async def remove_from_queue(task_id: str):
    global conversion_queue
    tasks_to_keep = [task for task in conversion_queue if str(task.task_id) != str(task_id)]
    removed_count = len(conversion_queue) - len(tasks_to_keep)
    conversion_queue = tasks_to_keep

    # Also remove from task_status
    if task_id in task_status:
        del task_status[task_id]

    save_queue_state()
    return {"status": "removed", "count": removed_count}

@app.post("/api/remove-all-pending")
async def remove_all_pending():
    """Remove all pending tasks from the conversion queue"""
    global conversion_queue

    try:
        # Count pending tasks before removal
        pending_count = len([task for task in conversion_queue if task.status == 'pending'])

        # Remove all pending tasks
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
        # Find the task in the queue
        task_index = next((i for i, t in enumerate(conversion_queue) if t.task_id == task_id), -1)
        if task_index == -1:
            raise HTTPException(status_code=404, detail="Task not found")

        task = conversion_queue[task_index]

        # Reset task status to pending for retry
        task.status = "pending"
        task.progress = 0.0
        task.eta = None
        task.error = None

        save_queue_state()
        logger.info(f"Retrying conversion task {task_id} for scene: {task.scene.title}")
        return {"status": "retried"}
    except Exception as e:
        logger.error(f"Failed to retry conversion task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry conversion: {str(e)}")

async def process_conversion_queue():
    config = get_config()

    global queue_paused
    if queue_paused:
        return

    while conversion_queue and len(active_tasks) < config['max_concurrent_tasks']:
        task = conversion_queue[0]
        if task.status == "pending":
            active_tasks.add(task.task_id)
            task.status = "processing"
            asyncio.create_task(convert_video(task))
            clear_sse_cache()  # Force SSE update when task starts
            save_queue_state()

        await asyncio.sleep(0.1)

class FFmpegProgress:
    def __init__(self, total_duration: float):
        self.total_duration = total_duration
        self.current_time = 0.0
        self.progress = 0.0
        self.last_update = time.time()

    def parse_line(self, line: str):
        """Parse FFmpeg output line to extract progress information"""
        line = line.strip()
        current_time = time.time()

        # Only process progress updates every 0.5 seconds to reduce CPU load
        if current_time - self.last_update < 0.5:
            return False

        self.last_update = current_time

        # Try multiple FFmpeg progress output formats
        time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
        if time_match:
            time_str = time_match.group(1)
            self.current_time = self.parse_time_string(time_str)
            if self.total_duration > 0:
                self.progress = min((self.current_time / self.total_duration) * 100, 99.0)  # Cap at 99% until complete
            return True

        # Alternative format: time=00:01:23.45
        time_match = re.search(r'time=(\d{2}:\d{2}:\d{2}\.\d{2})', line)
        if time_match:
            time_str = time_match.group(1)
            self.current_time = self.parse_time_string(time_str)
            if self.total_duration > 0:
                self.progress = min((self.current_time / self.total_duration) * 100, 99.0)
            return True

        # Look for frame information as fallback
        frame_match = re.search(r'frame=\s*(\d+)', line)
        if frame_match and self.total_duration > 0:
            # Estimate progress based on frames if we don't have time info
            # This is a rough estimate - 30fps default
            estimated_duration = int(frame_match.group(1)) / 30
            self.progress = min((estimated_duration / self.total_duration) * 100, 99.0)
            return True

        return False

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
    # Handle migration from delete_original to overwrite_original
    overwrite_original = config.get('overwrite_original', True)
    if 'delete_original' in config and 'overwrite_original' not in config:
        overwrite_original = config.get('delete_original', True)
    video_settings = config['video_settings']

    try:
        # Use the first file from the scene
        if not task.scene.files or len(task.scene.files) == 0:
            raise Exception("No files found in scene")

        scene_file = task.scene.files[0]
        input_file = scene_file.path

        original_extension = os.path.splitext(input_file)[1].lower()
        new_extension = f".{video_settings['container']}"

        # Determine output filename based on overwrite settings and extension match
        if overwrite_original and original_extension == new_extension:
            # Same extension - overwrite the original file
            base_name = os.path.splitext(input_file)[0]
            final_output = input_file  # Will overwrite original
            temp_output = f"{base_name}.converting.{video_settings['container']}"
        elif not overwrite_original:
            # Don't overwrite - find available name
            base_name = os.path.splitext(input_file)[0]
            final_output = await find_available_filename(base_name, video_settings['container'])
            temp_output = f"{final_output}.converting"
        else:
            # Different extensions with overwrite enabled
            base_name = os.path.splitext(input_file)[0]
            final_output = await find_available_filename(base_name, video_settings['container'])
            temp_output = f"{final_output}.converting"

        # Get file duration for progress calculation
        file_duration = scene_file.duration or 0
        progress_tracker = FFmpegProgress(file_duration)

        logger.info(f"Starting conversion for {input_file}")
        # Build FFmpeg command with progress output
        ffmpeg_cmd = build_ffmpeg_command(input_file, temp_output, video_settings)

        # Write to log
        with open(task.log_file, 'a') as log:
            log.write(f"Starting conversion: {input_file} -> {final_output}\n")
            log.write(f"File duration: {file_duration} seconds\n")
            log.write(f"FFmpeg command: {ffmpeg_cmd}\n")
            log.write(f"Overwrite original: {overwrite_original}\n")
            log.write("-" * 80 + "\n")

        # Start time for ETA calculation
        start_time = time.time()

        # Run FFmpeg and capture progress
        process = await asyncio.create_subprocess_shell(
            ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid
        )

        # Read stderr line by line to capture progress
        while True:
            line_bytes = await process.stderr.readline()
            if not line_bytes:
                break

            # Check if process has finished
            if process.returncode is not None:
                # Process finished, break out of the loop
                break

            line = line_bytes.decode('utf-8', errors='ignore').strip()

            # Write to log
            with open(task.log_file, 'a') as log:
                log.write(line + '\n')

            # Parse progress
            if progress_tracker.parse_line(line):
                task.progress = progress_tracker.progress

                # Calculate ETA
                if task.progress > 0:
                    elapsed_time = time.time() - start_time
                    if task.progress < 100:  # Don't calculate ETA if we're at 100%
                        total_estimated_time = elapsed_time / (task.progress / 100)
                        remaining_time = total_estimated_time - elapsed_time
                        task.eta = max(0, remaining_time)  # Ensure ETA is not negative
                    else:
                        task.eta = 0

                logger.debug(f"Progress update: {task.progress:.1f}%, ETA: {task.eta:.1f}s, Current time: {progress_tracker.current_time:.1f}s")

            # Check if task was cancelled
            if task.status == "cancelled":
                process.terminate()
                break

            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

        # Capture any remaining output (though we should have read most of it)
        remaining_stdout, remaining_stderr = await process.communicate()

        # Decode the remaining output
        stdout = remaining_stdout.decode('utf-8', errors='ignore') if remaining_stdout else ""
        stderr = remaining_stderr.decode('utf-8', errors='ignore') if remaining_stderr else ""

        # Final log entry
        with open(task.log_file, 'a') as log:
            log.write("-" * 80 + "\n")
            log.write(f"FFmpeg process completed with return code: {process.returncode}\n")
            log.write(f"Overwrite original setting: {overwrite_original}\n")
            log.write(f"Original file exists: {os.path.exists(input_file) if input_file else 'N/A'}\n")

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

            # Handle different scenarios based on overwrite settings and extension match
            if overwrite_original and original_extension == new_extension:
                # Same extension - replace original file directly
                if os.path.exists(final_output):  # This is the original file
                    os.remove(final_output)  # Delete original
                os.rename(temp_output, final_output)

                # Only trigger scan, no Stash DB update needed since path is the same
                await trigger_stash_scan(final_output)

            elif overwrite_original and original_extension != new_extension:
                # Different extensions with overwrite enabled
                os.rename(temp_output, final_output)

                # Update Stash and only delete original after successful update
                try:
                    await update_stash_file(task.scene.id, scene_file.id, final_output, overwrite_original)

                    # Delete original only after successful Stash update
                    if overwrite_original and os.path.exists(input_file):
                        os.remove(input_file)
                        logger.info(f"Deleted original file: {input_file}")

                except Exception as stash_error:
                    # If Stash update fails, mark task for retry (Stash update only)
                    task.status = "error"
                    task.error = f"Stash update failed: {str(stash_error)}"
                    logger.error(f"Stash update failed for {final_output}: {stash_error}")
                    save_queue_state()
                    return

            else:
                # No overwrite - add as new file to scene
                os.rename(temp_output, final_output)

                # Add as new file to Stash scene
                await add_file_to_scene(task.scene.id, final_output, overwrite_original)

            # Delete original file if overwrite is enabled
            # Only delete if overwrite is enabled and we haven't already deleted it
            if overwrite_original and original_extension == new_extension:
                # Already deleted above in the same extension case
                pass
            else:
                original_file_exists = os.path.exists(input_file)
                if original_file_exists and overwrite_original:
                    os.remove(input_file)
                    logger.info(f"Deleted original file: {input_file}")
                elif not overwrite_original:
                    logger.info(f"Original file preserved (no overwrite): {input_file}")

            task.status = "completed"
            task.output_file = final_output
            task.progress = 100.0
            task.eta = 0

            logger.info(f"Successfully converted {input_file} to {final_output}")
            save_queue_state()  # Save on successful completion
        else:
            task.status = "error"
            task.error = f"FFmpeg failed with return code {process.returncode}"
            logger.error(f"FFmpeg conversion failed for {input_file}")
    except Exception as e:
        task.status = "error"
        task.error = str(e)
        logger.error(f"Conversion failed for {task.scene.files[0].path if task.scene.files else 'unknown'}: {e}")
        logger.info(f"Overwrite original setting: {overwrite_original}")

        # Clean up temporary file if it exists
        if 'temp_output' in locals() and os.path.exists(temp_output):
            try:
                logger.info(f"Cleaning up temporary file: {temp_output}")
                os.remove(temp_output)
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file {temp_output}: {cleanup_error}")

        save_queue_state()  # Save on error

def build_ffmpeg_command(input_file: str, output_file: str, settings: Dict[str, Any]) -> str:
    framerate_option = f"-r {settings['framerate']}" if settings.get('framerate') else ""

    # Use progress reporting and verbose output for better progress tracking
    cmd = f"""ffmpeg -y -hide_banner -loglevel verbose -i "{input_file}" -filter_complex "scale=ceil(iw*min(1\,min({settings['width']}/iw\,{settings['height']}/ih))/2)*2:-2" -c:v libx264 {framerate_option} -crf 28 -c:a aac -b:v {settings['bitrate']} -maxrate {settings['bitrate']} -bufsize {settings['buffer_size']} -f {settings['container']} "{output_file}" """

    logger.debug(f"FFmpeg command: {cmd}")

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

async def trigger_stash_scan(file_path: str):
    """Trigger Stash metadata scan for a file without updating database"""
    config = get_config()
    path_mappings = config.get('path_mappings', [])

    # Convert the host path back to Docker path for Stash
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

    # First, ensure the file is scanned into Stash
    await trigger_stash_scan(new_file_path)

    # Then use sceneAssignFile to add it to the scene
    assign_mutation = """
    mutation SceneAssignFile($scene_id: ID!, $file_id: ID!) {
      sceneAssignFile(input: { scene_id: $scene_id, file_id: $file_id })
    }
    """

    # We need to get the file ID after scanning
    find_file_query = """
    query FindFileByPath($path: String!) {
      findFiles(path: $path) {
        files {
          id
        }
      }
    }
    """

    try:
        file_result = await stash_request(find_file_query, {"path": docker_path})
        if file_result['data']['findFiles']['files']:
            file_id = file_result['data']['findFiles']['files'][0]['id']
            await stash_request(assign_mutation, {"scene_id": scene_id, "file_id": file_id})
            logger.info(f"Added file {new_basename} to scene {scene_id}")
    except Exception as e:
        logger.error(f"Failed to add file to scene: {e}")
        raise Exception(f"Failed to add file to scene: {str(e)}")

async def update_stash_file(scene_id: str, file_id: str, new_file_path: str, overwrite_original: bool):
    """Update Stash with the new file information"""
    config = get_config()
    path_mappings = config.get('path_mappings', [])

    # Convert the host path back to Docker path for Stash
    docker_path = apply_path_mappings(new_file_path, path_mappings)

    # Ensure the log file is created and writable
    #log_dir = os.path.dirname(task.log_file)
    #os.makedirs(log_dir, exist_ok=True)
    new_basename = os.path.basename(docker_path)

    # Use execSQL mutation to directly update the files table
    import time
    updated_at = int(time.time())

    # Format the SQL string directly (be careful with SQL injection - we control the values)
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

    # Trigger metadata scan to update file properties (only for overwrite mode)
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
            # Don't fail the entire process if scan fails, as the DB update was successful

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

                # Create serializable conversion status
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

                # Only send if data has changed
                if last_sse_data != status_data:
                    last_sse_data = status_data
                    yield f"data: {json.dumps(status_data)}\n\n"
                else:
                    # Send a keep-alive ping every 30 seconds
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

# Runtime queue state (not persisted)
# Initialize as paused by default
queue_paused = True

# Function to clear SSE cache when queue changes
def clear_sse_cache():
    global last_sse_data
    last_sse_data = None

def convert_bitrate_to_bps(bitrate_str: str) -> int:
    """Convert bitrate string (e.g., '1000k') to bits per second"""
    multipliers = {'k': 1000, 'm': 1000000, 'g': 1000000000}
    if bitrate_str and bitrate_str[-1].lower() in multipliers:
        return int(bitrate_str[:-1]) * multipliers[bitrate_str[-1].lower()]
    return int(bitrate_str) if bitrate_str else 0

if __name__ == "__main__":
    uvicorn.run("stash_shrink:app", host="0.0.0.0", port=9899, reload=True)
