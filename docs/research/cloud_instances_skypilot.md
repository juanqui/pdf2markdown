# SkyPilot Integration for Dynamic pdf2markdown Processing

## Architecture Overview

Your pdf2markdown application can leverage SkyPilot to automatically provision GPU instances, deploy vision-language models like Qwen2.5-VL, process document batches, and tear down resources. This architecture provides cost-effective scaling with automatic cleanup.

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│ PDF Processing  │───▶│   SkyPilot   │───▶│  DataCrunch     │
│ Application     │    │   Orchestrator│    │  GPU Instance   │
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                       │                    │
         │                       │                    │
    ┌────▼─────┐           ┌─────▼─────┐      ┌──────▼──────┐
    │ PDF Queue│           │Task Config│      │Qwen2.5-VL + │
    │& Results │           │   YAML    │      │   SGLang    │
    └──────────┘           └───────────┘      └─────────────┘
```

## Core Implementation

### 1. Project Structure

```
pdf_processor/
├── src/
│   ├── pdf_processor/
│   │   ├── __init__.py
│   │   ├── core.py              # Main application logic
│   │   ├── skypilot_manager.py  # SkyPilot integration
│   │   ├── pdf_queue.py         # Batch management
│   │   └── config.py            # Configuration
│   └── skypilot_tasks/
│       ├── qwen_vl_setup.yaml   # SkyPilot task definition
│       └── pdf_processing.py    # Remote processing script
├── requirements.txt
└── config.yaml
```

### 2. SkyPilot Manager Class

```python
# src/pdf_processor/skypilot_manager.py
import asyncio
import tempfile
import os
from typing import List, Dict, Optional
from pathlib import Path
import yaml
import sky
from sky import Task, Resources, Storage

class SkyPilotManager:
    def __init__(self, config: Dict):
        self.config = config
        self.cluster_name = None
        self.task_yaml_path = config['skypilot']['task_yaml_path']
        
    async def provision_cluster(self, pdf_batch: List[str]) -> str:
        """Provision a cluster for processing PDF batch."""
        
        # Generate unique cluster name
        import uuid
        self.cluster_name = f"pdf-proc-{uuid.uuid4().hex[:8]}"
        
        # Load and customize task configuration
        task_config = self._prepare_task_config(pdf_batch)
        
        # Create SkyPilot task
        task = Task.from_yaml(self.task_yaml_path)
        task.update_envs({'BATCH_SIZE': str(len(pdf_batch))})
        
        # Set resource requirements
        task.set_resources(Resources(
            cloud='datacrunch',  # Specify DataCrunch
            instance_type=self.config['skypilot']['instance_type'],
            accelerators={'V100': 1},  # Adjust based on model requirements
            disk_size=50,  # GB
            use_spot=self.config.get('use_spot_instances', True)
        ))
        
        # Add storage for PDFs and results
        storage = Storage(
            name=f'pdf-storage-{self.cluster_name}',
            source='./pdf_inputs',  # Local directory with PDFs
            mount='/tmp/pdf_inputs'
        )
        task.set_storage_mounts({'pdf_inputs': storage})
        
        try:
            # Launch cluster
            await asyncio.to_thread(sky.launch, task, cluster_name=self.cluster_name)
            return self.cluster_name
            
        except Exception as e:
            await self.cleanup_cluster()
            raise RuntimeError(f"Failed to provision cluster: {e}")
    
    async def process_pdfs(self, pdf_files: List[str]) -> List[Dict]:
        """Process PDFs using the provisioned cluster."""
        
        if not self.cluster_name:
            raise ValueError("No cluster provisioned")
        
        # Upload PDFs to cluster
        await self._upload_pdfs(pdf_files)
        
        # Execute processing
        processing_script = """
python /tmp/pdf_processing.py \
    --input_dir /tmp/pdf_inputs \
    --output_dir /tmp/results \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --batch_size 4
"""
        
        try:
            # Run processing on cluster
            await asyncio.to_thread(
                sky.exec, 
                task=processing_script,
                cluster_name=self.cluster_name,
                stream_logs=True
            )
            
            # Download results
            results = await self._download_results()
            return results
            
        except Exception as e:
            raise RuntimeError(f"Processing failed: {e}")
    
    async def cleanup_cluster(self):
        """Cleanup cluster resources."""
        if self.cluster_name:
            try:
                await asyncio.to_thread(sky.down, self.cluster_name)
                print(f"Cleaned up cluster: {self.cluster_name}")
            except Exception as e:
                print(f"Cleanup warning: {e}")
            finally:
                self.cluster_name = None
    
    def _prepare_task_config(self, pdf_batch: List[str]) -> Dict:
        """Prepare task configuration for current batch."""
        
        # Create temporary directory for this batch
        batch_dir = Path(f"./pdf_inputs/batch_{len(pdf_batch)}")
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy PDFs to batch directory
        for pdf_file in pdf_batch:
            import shutil
            shutil.copy2(pdf_file, batch_dir)
        
        return {
            'batch_id': batch_dir.name,
            'pdf_count': len(pdf_batch),
            'model_config': self.config['model']
        }
    
    async def _upload_pdfs(self, pdf_files: List[str]):
        """Upload PDF files to cluster storage."""
        # Files are automatically synced via SkyPilot Storage
        pass
    
    async def _download_results(self) -> List[Dict]:
        """Download processing results from cluster."""
        
        # Download results directory
        download_cmd = "sky download {cluster_name} /tmp/results ./results/"
        
        try:
            await asyncio.to_thread(
                sky.exec,
                task=download_cmd.format(cluster_name=self.cluster_name),
                cluster_name=self.cluster_name
            )
            
            # Parse results
            results_dir = Path('./results')
            results = []
            
            for result_file in results_dir.glob('*.json'):
                import json
                with open(result_file) as f:
                    results.append(json.load(f))
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to download results: {e}")

```

### 3. PDF Queue Management

```python
# src/pdf_processor/pdf_queue.py
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import time

@dataclass
class ProcessingJob:
    id: str
    pdf_files: List[str]
    status: str  # 'queued', 'processing', 'completed', 'failed'
    created_at: float
    cluster_name: Optional[str] = None
    results: Optional[List[Dict]] = None
    error: Optional[str] = None

class PDFQueue:
    def __init__(self, config: Dict):
        self.config = config
        self.jobs: Dict[str, ProcessingJob] = {}
        self.queue: List[str] = []
        self.max_batch_size = config.get('max_batch_size', 10)
        self.batch_timeout = config.get('batch_timeout', 300)  # 5 minutes
        
    def add_pdfs(self, pdf_files: List[str]) -> str:
        """Add PDFs to processing queue."""
        import uuid
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        job = ProcessingJob(
            id=job_id,
            pdf_files=pdf_files,
            status='queued',
            created_at=time.time()
        )
        
        self.jobs[job_id] = job
        self.queue.append(job_id)
        
        return job_id
    
    def get_ready_batch(self) -> Optional[List[str]]:
        """Get batch of jobs ready for processing."""
        if not self.queue:
            return None
        
        # Check if we have enough jobs or timeout reached
        oldest_job = self.jobs[self.queue[0]]
        time_since_oldest = time.time() - oldest_job.created_at
        
        if len(self.queue) >= self.max_batch_size or time_since_oldest > self.batch_timeout:
            # Return batch
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]
            return batch
        
        return None
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job status and results."""
        return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status."""
        if job_id in self.jobs:
            self.jobs[job_id].status = status
            for key, value in kwargs.items():
                setattr(self.jobs[job_id], key, value)
```

### 4. Main Application Integration

```python
# src/pdf_processor/core.py
import asyncio
from typing import List, Dict, Optional
from pathlib import Path
import yaml
import logging

from .skypilot_manager import SkyPilotManager
from .pdf_queue import PDFQueue, ProcessingJob

class PDFProcessor:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.skypilot_manager = SkyPilotManager(self.config)
        self.pdf_queue = PDFQueue(self.config)
        self.processing_active = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def submit_pdfs(self, pdf_files: List[str]) -> str:
        """Submit PDFs for processing."""
        
        # Validate PDF files exist
        for pdf_file in pdf_files:
            if not Path(pdf_file).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_file}")
        
        job_id = self.pdf_queue.add_pdfs(pdf_files)
        self.logger.info(f"Submitted job {job_id} with {len(pdf_files)} PDFs")
        
        # Trigger processing check
        asyncio.create_task(self._check_and_process())
        
        return job_id
    
    async def get_results(self, job_id: str) -> Optional[ProcessingJob]:
        """Get processing results for a job."""
        return self.pdf_queue.get_job_status(job_id)
    
    async def _check_and_process(self):
        """Check queue and process batches if ready."""
        
        if self.processing_active:
            return
        
        batch_job_ids = self.pdf_queue.get_ready_batch()
        if not batch_job_ids:
            return
        
        self.processing_active = True
        
        try:
            await self._process_batch(batch_job_ids)
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Mark jobs as failed
            for job_id in batch_job_ids:
                self.pdf_queue.update_job_status(job_id, 'failed', error=str(e))
        finally:
            self.processing_active = False
            
            # Check if more batches are ready
            if self.pdf_queue.queue:
                asyncio.create_task(self._check_and_process())
    
    async def _process_batch(self, job_ids: List[str]):
        """Process a batch of jobs."""
        
        self.logger.info(f"Processing batch with {len(job_ids)} jobs")
        
        # Collect all PDF files from batch
        all_pdfs = []
        for job_id in job_ids:
            job = self.pdf_queue.get_job_status(job_id)
            all_pdfs.extend(job.pdf_files)
            self.pdf_queue.update_job_status(job_id, 'processing')
        
        try:
            # Provision cluster
            cluster_name = await self.skypilot_manager.provision_cluster(all_pdfs)
            
            # Update jobs with cluster info
            for job_id in job_ids:
                self.pdf_queue.update_job_status(job_id, 'processing', cluster_name=cluster_name)
            
            # Process PDFs
            results = await self.skypilot_manager.process_pdfs(all_pdfs)
            
            # Distribute results back to jobs
            await self._distribute_results(job_ids, results)
            
        finally:
            # Always cleanup cluster
            await self.skypilot_manager.cleanup_cluster()
    
    async def _distribute_results(self, job_ids: List[str], results: List[Dict]):
        """Distribute results back to individual jobs."""
        
        # Create mapping of PDF files to results
        pdf_to_result = {result['pdf_file']: result for result in results}
        
        for job_id in job_ids:
            job = self.pdf_queue.get_job_status(job_id)
            job_results = []
            
            for pdf_file in job.pdf_files:
                if pdf_file in pdf_to_result:
                    job_results.append(pdf_to_result[pdf_file])
            
            self.pdf_queue.update_job_status(
                job_id, 
                'completed', 
                results=job_results
            )
            
            self.logger.info(f"Job {job_id} completed with {len(job_results)} results")

# Usage example
async def main():
    processor = PDFProcessor()
    
    # Submit PDFs for processing
    job_id = await processor.submit_pdfs([
        './documents/report1.pdf',
        './documents/report2.pdf',
        './documents/manual.pdf'
    ])
    
    # Poll for results
    while True:
        job = await processor.get_results(job_id)
        if job.status == 'completed':
            print(f"Processing complete! Results: {job.results}")
            break
        elif job.status == 'failed':
            print(f"Processing failed: {job.error}")
            break
        else:
            print(f"Status: {job.status}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
```

## SkyPilot Task Configuration

### 5. Qwen2.5-VL Setup Task

```yaml
# src/skypilot_tasks/qwen_vl_setup.yaml

name: pdf2markdown-qwen

resources:
  cloud: datacrunch
  accelerators: V100:1
  disk_size: 50

file_mounts:
  /tmp/pdf_processing.py: ./src/skypilot_tasks/pdf_processing.py

setup: |
  # Install system dependencies
  sudo apt-get update
  sudo apt-get install -y poppler-utils tesseract-ocr
  
  # Install Python dependencies
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install sglang[all]
  pip install transformers accelerate
  pip install pdf2image pytesseract
  pip install pillow markdown

run: |
  # Start SGLang server with Qwen2.5-VL
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 &
  
  # Wait for server to start
  sleep 30
  
  # Keep server running
  wait
```

### 6. PDF Processing Script

```python
# src/skypilot_tasks/pdf_processing.py
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import requests
import base64
from pdf2image import convert_from_path
from PIL import Image
import io

class PDFToMarkdownProcessor:
    def __init__(self, sglang_url: str = "http://localhost:30000"):
        self.sglang_url = sglang_url
        
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process a single PDF to markdown."""
        
        print(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=150)
        
        markdown_content = ""
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            
            # Convert image to base64
            image_b64 = self._image_to_base64(image)
            
            # Generate markdown for this page
            page_markdown = self._image_to_markdown(image_b64, page_num=i+1)
            
            markdown_content += f"\n\n<!-- Page {i+1} -->\n\n"
            markdown_content += page_markdown
        
        return {
            'pdf_file': pdf_path,
            'markdown_content': markdown_content,
            'page_count': len(images),
            'status': 'success'
        }
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')
    
    def _image_to_markdown(self, image_b64: str, page_num: int) -> str:
        """Convert image to markdown using Qwen2.5-VL."""
        
        prompt = """Analyze this document page and convert it to clean, well-formatted Markdown.

Instructions:
- Preserve the document structure with appropriate headers
- Convert tables to Markdown table format
- Extract all text content accurately
- Maintain logical flow and formatting
- Use proper Markdown syntax for lists, emphasis, links, etc.
- If there are figures or charts, describe them briefly

Document page to convert:"""

        payload = {
            "text": prompt,
            "image_data": image_b64,
            "sampling_params": {
                "temperature": 0.1,
                "max_tokens": 2048,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{self.sglang_url}/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('text', f'[Error processing page {page_num}]')
            
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            return f"[Error processing page {page_num}: {e}]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing PDF files')
    parser.add_argument('--output_dir', required=True, help='Directory for output results')
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor
    processor = PDFToMarkdownProcessor()
    
    # Process all PDFs in input directory
    pdf_files = list(Path(args.input_dir).glob('*.pdf'))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    results = []
    
    for pdf_file in pdf_files:
        try:
            result = processor.process_pdf(str(pdf_file))
            results.append(result)
            
            # Save individual result
            output_file = Path(args.output_dir) / f"{pdf_file.stem}_result.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Save markdown file
            markdown_file = Path(args.output_dir) / f"{pdf_file.stem}.md"
            with open(markdown_file, 'w') as f:
                f.write(result['markdown_content'])
                
        except Exception as e:
            error_result = {
                'pdf_file': str(pdf_file),
                'error': str(e),
                'status': 'failed'
            }
            results.append(error_result)
            print(f"Error processing {pdf_file}: {e}")
    
    # Save summary results
    summary_file = Path(args.output_dir) / 'processing_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'total_files': len(pdf_files),
            'successful': len([r for r in results if r.get('status') == 'success']),
            'failed': len([r for r in results if r.get('status') == 'failed']),
            'results': results
        }, f, indent=2)
    
    print(f"Processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
```

## Configuration

### 7. Configuration File

```yaml
# config.yaml
skypilot:
  task_yaml_path: './src/skypilot_tasks/qwen_vl_setup.yaml'
  instance_type: 'V100'  # Adjust based on DataCrunch offerings
  
model:
  name: 'Qwen/Qwen2.5-VL-7B-Instruct'
  max_tokens: 2048
  temperature: 0.1

# Batch processing settings
max_batch_size: 5          # Max PDFs per batch
batch_timeout: 300         # 5 minutes
use_spot_instances: true   # Use spot instances for cost savings

# DataCrunch specific settings
datacrunch:
  region: 'FIN-01'  # Adjust based on preference
  pricing_model: 'dynamic'  # or 'fixed'
```

## Advanced Features

### 8. Real-time Progress Tracking

```python
# Extension for real-time progress tracking
import websockets
import json
from typing import Set

class ProgressTracker:
    def __init__(self):
        self.subscribers: Set[websockets.WebSocketServerProtocol] = set()
    
    async def subscribe(self, websocket):
        """Subscribe to progress updates."""
        self.subscribers.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.subscribers.remove(websocket)
    
    async def broadcast_update(self, job_id: str, progress: Dict):
        """Broadcast progress update to subscribers."""
        message = {
            'job_id': job_id,
            'progress': progress,
            'timestamp': time.time()
        }
        
        if self.subscribers:
            await asyncio.gather(
                *[ws.send(json.dumps(message)) for ws in self.subscribers],
                return_exceptions=True
            )
```

### 9. Cost Monitoring Integration

```python
# Extension for cost monitoring
class CostMonitor:
    def __init__(self, budget_limit: float = 100.0):
        self.budget_limit = budget_limit
        self.current_spend = 0.0
        
    async def track_cluster_cost(self, cluster_name: str):
        """Track cluster costs in real-time."""
        # Integrate with DataCrunch billing API
        # or estimate based on instance type and runtime
        pass
    
    def check_budget(self) -> bool:
        """Check if budget limit exceeded."""
        return self.current_spend < self.budget_limit
```

## Deployment and Usage

### 10. Quick Start Example

```python
from pdf_processor import PDFProcessor

# Initialize processor
processor = PDFProcessor('config.yaml')

# Process documents
job_id = await processor.submit_pdfs([
    './docs/research_paper.pdf',
    './docs/technical_manual.pdf'
])

# Get results
results = await processor.get_results(job_id)
print(f"Converted {len(results)} documents to Markdown")
```

This architecture provides:
- **Automatic scaling**: Spins up compute only when needed
- **Cost optimization**: Uses spot instances and automatic teardown
- **Batch efficiency**: Processes multiple PDFs per cluster
- **Error handling**: Robust failure recovery and cleanup
- **Progress tracking**: Real-time status updates
- **Provider flexibility**: Easy switching between cloud providers

The integration with SkyPilot handles all the complexity of instance provisioning, software installation, and resource cleanup, while your application focuses on the business logic of PDF processing.