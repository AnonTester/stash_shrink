import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid
import subprocess
import shutil

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

class Scene(BaseModel):
    id: str
    title: str
    details: Optional[str] = None
    path: str
    file: Dict[str, Any]

class ConversionTask(BaseModel):
    task_id: str
    scene: Scene
    status: str = "pending"  # pending, processing, completed, error
    progress: float = 0.0
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
        config = DEFAULT_CONFIG
        save_config(config)
    return config

async def stash_request(graphql_query: str, variables: dict = None):
    config = get_config()
    headers = {}
    if config.get('api_key'):
        headers['ApiKey'] = config['api_key']
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{config['stash_url']}/graphql",
            json={"query": graphql_query, "variables": variables},
            headers=headers
        )
        return response.json()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    config = get_config()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": config
    })

@app.get("/api/config")
async def get_config_api():
    return get_config()

@app.post("/api/config")
async def update_config(settings: Settings):
    save_config(settings.dict())
    return {"status": "ok"}

@app.post("/api/search")
async def search_scenes(search_params: SearchParams):
    # Build GraphQL query for Stash
    filter_conditions = []
    
    if search_params.max_width:
        filter_conditions.append(f'width: {{value: {search_params.max_width}, modifier: GREATER_THAN}}')
    if search_params.max_height:
        filter_conditions.append(f'height: {{value: {search_params.max_height}, modifier: GREATER_THAN}}')
    if search_params.max_bitrate:
        # Convert bitrate string to bits per second
        bitrate_value = convert_bitrate_to_bps(search_params.max_bitrate)
        filter_conditions.append(f'bit_rate: {{value: {bitrate_value}, modifier: GREATER_THAN}}')
    if search_params.max_framerate:
        filter_conditions.append(f'frame_rate: {{value: {search_params.max_framerate}, modifier: GREATER_THAN}}')
    if search_params.codec:
        filter_conditions.append(f'video_codec: {{value: "{search_params.codec}", modifier: NOT_EQUALS}}')
    if search_params.path:
        filter_conditions.append(f'path: {{value: "{search_params.path}", modifier: INCLUDES}}')
    
    filter_str = ", ".join(filter_conditions)
    
    query = f"""
    query SearchScenes {{
      scenes(filter: {{ {filter_str} }}) {{
        id
        title
        details
        path
        file {{
          size
          duration
          video_codec
          width
          height
          bit_rate
          frame_rate
        }}
      }}
    }}
    """
    
    result = await stash_request(query)
    if 'errors' in result:
        raise HTTPException(status_code=400, detail=result['errors'])
    
    scenes = []
    for scene_data in result['data']['scenes']:
        scenes.append(Scene(**scene_data))
    
    return {"scenes": scenes}

@app.post("/api/queue-conversion")
async def queue_conversion(scene_ids: List[str]):
    config = get_config()
    
    # Get scene details from Stash
    scenes_query = """
    query GetScenes($ids: [ID!]) {
      scenes(ids: $ids) {
        id
        title
        details
        path
        file {
          size
          duration
          video_codec
          width
          height
          bit_rate
          frame_rate
        }
      }
    }
    """
    
    result = await stash_request(scenes_query, {"ids": scene_ids})
    if 'errors' in result:
        raise HTTPException(status_code=400, detail=result['errors'])
    
    for scene_data in result['data']['scenes']:
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
    
    # Start processing if not already running
    if len(active_tasks) < config['max_concurrent_tasks']:
        asyncio.create_task(process_conversion_queue())
    
    return {"status": "queued", "count": len(scene_ids)}

@app.get("/api/conversion-status")
async def conversion_status():
    return {
        "queue": [task.dict() for task in conversion_queue],
        "active": list(active_tasks),
        "completed": [task.dict() for task in conversion_queue if task.status in ["completed", "error"]]
    }

@app.post("/api/cancel-conversion/{task_id}")
async def cancel_conversion(task_id: str):
    if task_id in active_tasks:
        # This would need proper process termination in a real implementation
        active_tasks.remove(task_id)
    
    for i, task in enumerate(conversion_queue):
        if task.task_id == task_id:
            conversion_queue.pop(i)
            task_status.pop(task_id, None)
            break
    
    return {"status": "cancelled"}

@app.post("/api/clear-completed")
async def clear_completed():
    global conversion_queue
    conversion_queue = [task for task in conversion_queue if task.status not in ["completed", "error"]]
    return {"status": "cleared"}

async def process_conversion_queue():
    config = get_config()
    
    while conversion_queue and len(active_tasks) < config['max_concurrent_tasks']:
        task = conversion_queue[0]
        if task.status == "pending":
            active_tasks.add(task.task_id)
            task.status = "processing"
            asyncio.create_task(convert_video(task))
        
        # Remove from queue if completed or error
        if task.status in ["completed", "error"]:
            conversion_queue.pop(0)
            active_tasks.discard(task.task_id)
        
        await asyncio.sleep(0.1)

async def convert_video(task: ConversionTask):
    config = get_config()
    video_settings = config['video_settings']
    
    try:
        input_file = task.scene.path
        base_name = os.path.splitext(input_file)[0]
        temp_output = f"{base_name}.converting.{video_settings['container']}"
        final_output = await find_available_filename(base_name, video_settings['container'])
        
        # Build FFmpeg command
        ffmpeg_cmd = build_ffmpeg_command(input_file, temp_output, video_settings)
        
        # Write to log
        with open(task.log_file, 'a') as log:
            log.write(f"Starting conversion: {input_file} -> {final_output}\n")
            log.write(f"FFmpeg command: {ffmpeg_cmd}\n")
        
        # Run FFmpeg
        process = await asyncio.create_subprocess_shell(
            ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Monitor progress (simplified - in reality you'd parse FFmpeg output)
        while process.returncode is None:
            await asyncio.sleep(1)
            # Update progress (this is simplified)
            task.progress += 1  # Would need proper progress calculation
        
        stdout, stderr = await process.communicate()
        
        with open(task.log_file, 'a') as log:
            log.write(stdout.decode())
            log.write(stderr.decode())
        
        if process.returncode == 0:
            # Rename temporary file to final name
            os.rename(temp_output, final_output)
            
            # Update Stash
            await update_stash_scene(task.scene.id, final_output)
            
            # Delete original file
            os.remove(input_file)
            
            task.status = "completed"
            task.output_file = final_output
        else:
            task.status = "error"
            task.error = f"FFmpeg failed with return code {process.returncode}"
    
    except Exception as e:
        task.status = "error"
        task.error = str(e)
        logger.error(f"Conversion failed for {task.scene.path}: {e}")

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

async def update_stash_scene(scene_id: str, new_file_path: str):
    # This would update the Stash scene with the new file
    # Implementation depends on Stash's GraphQL API
    pass

@app.get("/sse")
async def sse_endpoint(request: Request):
    async def event_generator():
        client_id = str(uuid.uuid4())
        sse_clients.add(client_id)
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                # Send conversion status
                status_data = {
                    "queue": [task.dict() for task in conversion_queue],
                    "active": list(active_tasks)
                }
                
                yield f"data: {json.dumps(status_data)}\n\n"
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
    """Convert bitrate string (e.g., '1000k') to bits per second"""
    multipliers = {'k': 1000, 'm': 1000000, 'g': 1000000000}
    if bitrate_str[-1].lower() in multipliers:
        return int(bitrate_str[:-1]) * multipliers[bitrate_str[-1].lower()]
    return int(bitrate_str)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9899, reload=True)
