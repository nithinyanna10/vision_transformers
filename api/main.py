#!/usr/bin/env python3
"""
REST API for Medical Vision Transformer
FastAPI-based API with model inference, batch processing, and health monitoring
"""

import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import base64
import io
import json
import logging

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our model components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model import create_model
from src.explainable_ai import GradCAM, AttentionVisualizer, LIMEExplainer
from src.model_optimization import ModelQuantizer, ONNXConverter, PerformanceBenchmark

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Vision Transformer API",
    description="Production-ready API for chest X-ray pneumonia detection using Vision Transformers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
config = None
device = None
model_loaded = False
inference_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_inference_time": 0.0,
    "total_inference_time": 0.0
}

# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    image_base64: str = Field(..., description="Base64 encoded image")
    include_explanations: bool = Field(False, description="Include XAI explanations")
    confidence_threshold: float = Field(0.5, description="Confidence threshold for predictions")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    images_base64: List[str] = Field(..., description="List of base64 encoded images")
    include_explanations: bool = Field(False, description="Include XAI explanations")
    confidence_threshold: float = Field(0.5, description="Confidence threshold for predictions")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str = Field(..., description="Predicted class (Normal/Pneumonia)")
    confidence: float = Field(..., description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    explanations: Optional[Dict[str, Any]] = Field(None, description="XAI explanations")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of images processed")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")
    average_time_per_image_ms: float = Field(..., description="Average time per image")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    inference_stats: Dict[str, Any] = Field(..., description="Inference statistics")

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str = Field(..., description="Type of model")
    model_size_mb: float = Field(..., description="Model size in MB")
    num_parameters: int = Field(..., description="Number of model parameters")
    input_shape: List[int] = Field(..., description="Expected input shape")
    classes: List[str] = Field(..., description="Output classes")
    config: Dict[str, Any] = Field(..., description="Model configuration")


def load_model():
    """Load the trained model"""
    global model, config, device, model_loaded
    
    try:
        logger.info("Loading model...")
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create model
        model = create_model(config)
        
        # Load trained weights
        model_path = Path(__file__).parent.parent / "models" / "best_model.pth"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded successfully")
        else:
            logger.warning("No trained model found, using random weights")
        
        model = model.to(device)
        model.eval()
        model_loaded = True
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False
        raise e


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image for model inference"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image_size = config['model']['img_size']
        image = image.resize((image_size, image_size))
        
        # Convert to tensor
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image_tensor = (image_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        
        return image_tensor.unsqueeze(0).to(device)
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")


def predict_single(image_tensor: torch.Tensor, include_explanations: bool = False) -> Dict[str, Any]:
    """Make prediction for a single image"""
    global inference_stats
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # Make prediction
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = outputs.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Get class names
        class_names = ['Normal', 'Pneumonia']
        predicted_class = class_names[predicted_class_idx]
        
        # Calculate probabilities for all classes
        class_probabilities = {
            class_names[i]: probabilities[0, i].item() 
            for i in range(len(class_names))
        }
        
        inference_time = (time.time() - start_time) * 1000
        
        # Update statistics
        inference_stats["total_requests"] += 1
        inference_stats["successful_requests"] += 1
        inference_stats["total_inference_time"] += inference_time
        inference_stats["average_inference_time"] = (
            inference_stats["total_inference_time"] / inference_stats["successful_requests"]
        )
        
        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": class_probabilities,
            "inference_time_ms": inference_time
        }
        
        # Add explanations if requested
        if include_explanations:
            try:
                explanations = generate_explanations(image_tensor, predicted_class_idx)
                result["explanations"] = explanations
            except Exception as e:
                logger.warning(f"Failed to generate explanations: {e}")
                result["explanations"] = {"error": str(e)}
        
        return result
        
    except Exception as e:
        inference_stats["total_requests"] += 1
        inference_stats["failed_requests"] += 1
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def generate_explanations(image_tensor: torch.Tensor, predicted_class: int) -> Dict[str, Any]:
    """Generate XAI explanations for the prediction"""
    explanations = {}
    
    try:
        # Grad-CAM explanation
        gradcam = GradCAM(model, target_layer_name="transformer_blocks.0.attn")
        cam = gradcam.generate_cam(image_tensor)
        explanations["gradcam"] = {
            "description": "Grad-CAM visualization showing important regions",
            "heatmap_shape": cam.shape,
            "max_attention": float(cam.max()),
            "min_attention": float(cam.min())
        }
    except Exception as e:
        explanations["gradcam"] = {"error": str(e)}
    
    try:
        # Attention visualization
        attn_viz = AttentionVisualizer(model)
        attention_map = attn_viz.visualize_attention_rollout(image_tensor)
        explanations["attention"] = {
            "description": "Vision Transformer attention visualization",
            "attention_shape": attention_map.shape,
            "max_attention": float(attention_map.max()),
            "min_attention": float(attention_map.min())
        }
    except Exception as e:
        explanations["attention"] = {"error": str(e)}
    
    return explanations


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    logger.info("Starting Medical Vision Transformer API...")
    load_model()
    logger.info("API startup completed!")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Vision Transformer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        device=str(device) if device else "unknown",
        uptime_seconds=uptime,
        inference_stats=inference_stats
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate model size
    model_size = 0
    num_parameters = 0
    for param in model.parameters():
        num_parameters += param.numel()
        model_size += param.numel() * param.element_size()
    
    model_size_mb = model_size / (1024 * 1024)
    
    return ModelInfoResponse(
        model_type="Vision Transformer",
        model_size_mb=model_size_mb,
        num_parameters=num_parameters,
        input_shape=[1, 3, config['model']['img_size'], config['model']['img_size']],
        classes=['Normal', 'Pneumonia'],
        config=config
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction for a single image"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        result = predict_single(image_tensor, request.include_explanations)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple images"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.images_base64) > 10:
        raise HTTPException(status_code=400, detail="Batch size too large (max 10 images)")
    
    start_time = time.time()
    predictions = []
    
    try:
        for image_base64 in request.images_base64:
            # Decode base64 image
            image_bytes = base64.b64decode(image_base64)
            
            # Preprocess image
            image_tensor = preprocess_image(image_bytes)
            
            # Make prediction
            result = predict_single(image_tensor, request.include_explanations)
            predictions.append(PredictionResponse(**result))
        
        total_time = (time.time() - start_time) * 1000
        average_time = total_time / len(predictions)
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            total_time_ms=total_time,
            average_time_per_image_ms=average_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...), include_explanations: bool = False):
    """Make prediction for uploaded file"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        result = predict_single(image_tensor, include_explanations)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"File prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "inference_stats": inference_stats,
        "model_loaded": model_loaded,
        "device": str(device) if device else "unknown"
    }


@app.post("/model/reload")
async def reload_model():
    """Reload the model"""
    try:
        load_model()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    # Set startup time
    app.state.start_time = time.time()
    
    # Run the API
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
