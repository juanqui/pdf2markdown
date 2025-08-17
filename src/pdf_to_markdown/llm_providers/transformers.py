"""Transformers-based LLM provider implementation for local model inference."""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from pdf_to_markdown.core import LLMConnectionError

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class TransformersLLMProvider(LLMProvider):
    """LLM provider for Hugging Face Transformers models with vision capabilities."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the Transformers provider with configuration.

        Args:
            config: Configuration dictionary with the following keys:
                - model_name (str): HuggingFace model name/path (e.g., "openbmb/MiniCPM-V-4")
                - device (str): Device to run on ("cuda", "cpu", "auto") - default: "auto"
                - torch_dtype (str): Data type for model ("float16", "bfloat16", "float32", "auto") - default: "auto"
                - trust_remote_code (bool): Whether to trust remote code - default: True
                - attn_implementation (str): Attention implementation ("sdpa", "flash_attention_2", "eager") - default: "sdpa"
                - max_new_tokens (int): Maximum tokens to generate - default: 4096
                - temperature (float): Temperature for generation - default: 0.1
                - do_sample (bool): Whether to use sampling - default: False
                - device_map (str): Device mapping strategy ("auto", "balanced", etc.) - default: "auto"
                - load_in_8bit (bool): Load model in 8-bit mode - default: False
                - load_in_4bit (bool): Load model in 4-bit mode - default: False
                - cache_dir (str): Directory to cache models - default: None
                - model_type (str): Model type hint ("vision", "image-text-to-text") - default: "auto"
                - use_chat_method (bool): Whether model has a .chat() method - default: False
                - processor_type (str): Type of processor ("auto", "processor", "tokenizer") - default: "auto"
        """
        super().__init__(config)
        
        # Check for required dependencies
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "transformers library is not installed. "
                "Install it with: pip install pdf-to-markdown[transformers]"
            )
        
        self.model_name = config.get("model_name")
        if not self.model_name:
            raise LLMConnectionError("model_name is required for TransformersLLMProvider")
        
        # Model loading configuration
        self.device = config.get("device", "auto")
        self.torch_dtype_str = config.get("torch_dtype", "auto")
        self.trust_remote_code = config.get("trust_remote_code", True)
        self.attn_implementation = config.get("attn_implementation", "sdpa")
        self.device_map = config.get("device_map", "auto")
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.load_in_4bit = config.get("load_in_4bit", False)
        self.cache_dir = config.get("cache_dir")
        
        # Generation configuration
        self.max_new_tokens = config.get("max_new_tokens", 4096)
        self.temperature = config.get("temperature", 0.1)
        self.do_sample = config.get("do_sample", False)
        
        # Model type hints
        self.model_type = config.get("model_type", "auto")
        self.use_chat_method = config.get("use_chat_method", False)
        self.processor_type = config.get("processor_type", "auto")
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._initialize_model()
        
        logger.info(
            f"Initialized TransformersLLMProvider with model={self.model_name}, "
            f"device={self.device}, dtype={self.torch_dtype_str}"
        )

    def _get_torch_dtype(self):
        """Convert string dtype to torch dtype."""
        if self.torch_dtype_str == "auto":
            return "auto"
        elif self.torch_dtype_str == "float16":
            return torch.float16
        elif self.torch_dtype_str == "bfloat16":
            return torch.bfloat16
        elif self.torch_dtype_str == "float32":
            return torch.float32
        else:
            logger.warning(f"Unknown torch_dtype: {self.torch_dtype_str}, using auto")
            return "auto"

    def _initialize_model(self):
        """Initialize the model and processor/tokenizer."""
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )
        
        torch_dtype = self._get_torch_dtype()
        
        # Common model loading arguments
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch_dtype,
            "cache_dir": self.cache_dir,
        }
        
        # Add device mapping
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        elif self.device != "auto":
            model_kwargs["device_map"] = self.device
            
        # Add attention implementation if not using quantization
        if not self.load_in_8bit and not self.load_in_4bit:
            model_kwargs["attn_implementation"] = self.attn_implementation
            
        # Add quantization options
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        try:
            # Try to load processor first (for multimodal models)
            if self.processor_type != "tokenizer":
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=self.trust_remote_code,
                        cache_dir=self.cache_dir,
                    )
                    logger.info(f"Loaded processor for {self.model_name}")
                except Exception as e:
                    logger.debug(f"Could not load processor: {e}")
                    
            # Load tokenizer if processor wasn't loaded or if specifically requested
            if self.processor is None or self.processor_type == "tokenizer":
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=self.trust_remote_code,
                        cache_dir=self.cache_dir,
                    )
                    logger.info(f"Loaded tokenizer for {self.model_name}")
                except Exception as e:
                    logger.warning(f"Could not load tokenizer: {e}")
            
            # Try different model loading strategies based on model type
            model_loaded = False
            
            # First try AutoModelForImageTextToText (for vision models)
            if not model_loaded and self.model_type in ["auto", "vision", "image-text-to-text"]:
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name, **model_kwargs
                    )
                    logger.info(f"Loaded model as AutoModelForImageTextToText")
                    model_loaded = True
                except Exception as e:
                    logger.debug(f"Could not load as AutoModelForImageTextToText: {e}")
            
            # Try AutoModelForCausalLM
            if not model_loaded and self.model_type in ["auto", "causal"]:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, **model_kwargs
                    )
                    logger.info(f"Loaded model as AutoModelForCausalLM")
                    model_loaded = True
                except Exception as e:
                    logger.debug(f"Could not load as AutoModelForCausalLM: {e}")
            
            # Finally try generic AutoModel
            if not model_loaded:
                try:
                    self.model = AutoModel.from_pretrained(
                        self.model_name, **model_kwargs
                    )
                    logger.info(f"Loaded model as AutoModel")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Could not load model: {e}")
                    raise LLMConnectionError(f"Failed to load model {self.model_name}: {e}")
            
            # Move to eval mode
            self.model.eval()
            
            # Move to specific device if not using device_map
            if not self.device_map and self.device != "auto":
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.cuda()
                elif self.device == "cpu":
                    self.model = self.model.cpu()
                    
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize model {self.model_name}: {e}")

    def _prepare_image(self, image_path: Optional[Path] = None, image_base64: Optional[str] = None) -> Image.Image:
        """Prepare image from file path or base64 string.
        
        Args:
            image_path: Path to image file
            image_base64: Base64 encoded image string
            
        Returns:
            PIL Image object
        """
        if image_path:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            return Image.open(image_path).convert("RGB")
        elif image_base64:
            image_bytes = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            raise ValueError("Either image_path or image_base64 must be provided")

    async def _generate_with_chat_method(self, image: Image.Image, prompt: str, **kwargs: Any) -> str:
        """Generate using model's .chat() method (for models like MiniCPM-V)."""
        msgs = [{"role": "user", "content": [image, prompt]}]
        
        generation_kwargs = {
            "msgs": msgs,
            "image": image,
            "tokenizer": self.tokenizer or self.processor.tokenizer,
            "sampling": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
        }
        
        # Filter out None values
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        
        # Generate
        response = self.model.chat(**generation_kwargs)
        return response

    async def _generate_with_processor(self, image: Image.Image, prompt: str, **kwargs: Any) -> str:
        """Generate using processor and standard generation (for most vision models)."""
        # Prepare messages in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template if available
        if hasattr(self.processor, "apply_chat_template"):
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            # Fallback to direct processing
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding=True,
            )
        
        # Move inputs to device
        if self.device != "auto" and self.device != "cpu":
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        elif hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "do_sample": self.do_sample,
        }
        
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode the output
        if hasattr(self.processor, "batch_decode"):
            # Extract only the generated tokens (skip input)
            generated_ids = output_ids[:, inputs.get("input_ids", output_ids).shape[1]:]
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
        elif hasattr(self.processor, "decode"):
            generated_ids = output_ids[0, inputs.get("input_ids", output_ids).shape[1]:]
            output_text = self.processor.decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        elif self.tokenizer:
            # Use tokenizer if processor doesn't have decode
            generated_ids = output_ids[:, inputs.get("input_ids", output_ids).shape[1]:]
            output_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
        else:
            raise LLMConnectionError("No method available to decode output")
        
        return output_text

    async def invoke_with_image(self, prompt: str, image_path: Path, **kwargs: Any) -> LLMResponse:
        """Invoke the LLM with a text prompt and an image.

        Args:
            prompt: Text prompt for the LLM
            image_path: Path to the image file
            **kwargs: Additional parameters for generation

        Returns:
            LLMResponse containing the generated text and metadata
        """
        logger.debug(f"Invoking Transformers LLM with image from {image_path}")
        
        try:
            # Prepare the image
            image = self._prepare_image(image_path=image_path)
            
            # Generate based on model type
            if self.use_chat_method and hasattr(self.model, "chat"):
                output_text = await self._generate_with_chat_method(image, prompt, **kwargs)
            elif self.processor is not None:
                output_text = await self._generate_with_processor(image, prompt, **kwargs)
            else:
                raise LLMConnectionError(
                    f"Model {self.model_name} does not support image input or no processor available"
                )
            
            return LLMResponse(
                content=output_text,
                model=self.model_name,
                metadata={
                    "provider": "transformers",
                    "device": str(self.device),
                    "dtype": str(self.torch_dtype_str),
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating with Transformers model: {e}")
            raise LLMConnectionError(f"Failed to generate: {e}")

    async def invoke_with_image_base64(
        self, prompt: str, image_base64: str, **kwargs: Any
    ) -> LLMResponse:
        """Invoke the LLM with a text prompt and a base64-encoded image.

        Args:
            prompt: Text prompt for the LLM
            image_base64: Base64-encoded image string
            **kwargs: Additional parameters for generation

        Returns:
            LLMResponse containing the generated text and metadata
        """
        logger.debug(f"Invoking Transformers LLM with base64 image")
        
        try:
            # Prepare the image
            image = self._prepare_image(image_base64=image_base64)
            
            # Generate based on model type
            if self.use_chat_method and hasattr(self.model, "chat"):
                output_text = await self._generate_with_chat_method(image, prompt, **kwargs)
            elif self.processor is not None:
                output_text = await self._generate_with_processor(image, prompt, **kwargs)
            else:
                raise LLMConnectionError(
                    f"Model {self.model_name} does not support image input or no processor available"
                )
            
            return LLMResponse(
                content=output_text,
                model=self.model_name,
                metadata={
                    "provider": "transformers",
                    "device": str(self.device),
                    "dtype": str(self.torch_dtype_str),
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating with Transformers model: {e}")
            raise LLMConnectionError(f"Failed to generate: {e}")

    async def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Invoke the LLM with a text prompt only.

        Args:
            prompt: Text prompt for the LLM
            **kwargs: Additional parameters for generation

        Returns:
            LLMResponse containing the generated text and metadata
        """
        logger.debug(f"Invoking Transformers LLM with text-only prompt")
        
        try:
            # For text-only generation
            if self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt")
            elif self.processor and hasattr(self.processor, "tokenizer"):
                inputs = self.processor.tokenizer(prompt, return_tensors="pt")
            else:
                raise LLMConnectionError("No tokenizer available for text-only generation")
            
            # Move to device
            if self.device != "auto" and self.device != "cpu":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            elif hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
                "do_sample": self.do_sample,
            }
            
            if self.do_sample:
                generation_kwargs["temperature"] = self.temperature
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
            
            # Decode
            if self.tokenizer:
                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                output_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0]
            elif self.processor and hasattr(self.processor, "tokenizer"):
                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                output_text = self.processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0]
            else:
                raise LLMConnectionError("No method available to decode output")
            
            return LLMResponse(
                content=output_text,
                model=self.model_name,
                metadata={
                    "provider": "transformers",
                    "device": str(self.device),
                    "dtype": str(self.torch_dtype_str),
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating with Transformers model: {e}")
            raise LLMConnectionError(f"Failed to generate: {e}")

    async def cleanup(self) -> None:
        """Cleanup model resources."""
        logger.info("Cleaning up Transformers LLM provider resources")
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
            torch.cuda.empty_cache()
        
        # Delete model and processor references
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        if self.tokenizer:
            del self.tokenizer
        
        self.model = None
        self.processor = None
        self.tokenizer = None

    def validate_config(self) -> bool:
        """Validate the provider configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        if not self.model_name:
            logger.error("Model name is missing")
            return False
        
        if self.temperature < 0 or self.temperature > 2:
            logger.error(f"Invalid temperature: {self.temperature}")
            return False
        
        if self.max_new_tokens < 1:
            logger.error(f"Invalid max_new_tokens: {self.max_new_tokens}")
            return False
        
        # Check if CUDA is available when requested
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, will fall back to CPU")
        
        return True