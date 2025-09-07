#!/usr/bin/env python3
"""
Deployment Script for Medical Vision Transformer
Handles model optimization, containerization, and deployment
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model_optimization import ModelQuantizer, ONNXConverter, PerformanceBenchmark
from src.model import create_model


class DeploymentManager:
    """Complete deployment management system"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.deployment_dir = Path("deployment")
        self.deployment_dir.mkdir(exist_ok=True)
        
        # Deployment results
        self.deployment_results = {
            "optimization": {},
            "containerization": {},
            "api_deployment": {},
            "performance": {}
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def optimize_model(self, model_path: str = "models/best_model.pth") -> Dict[str, Any]:
        """Optimize model for deployment"""
        print("🚀 Starting model optimization...")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return {"status": "failed", "error": "Model file not found"}
        
        # Load model
        model = create_model(self.config)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize optimizers
        quantizer = ModelQuantizer(model, self.config)
        onnx_converter = ONNXConverter(model, self.config)
        benchmark = PerformanceBenchmark(self.config)
        
        optimization_results = {}
        
        try:
            # Dynamic quantization
            print("🔢 Applying dynamic quantization...")
            quantized_path = self.deployment_dir / "model_quantized_dynamic.pth"
            dynamic_results = quantizer.quantize_dynamic(model_path, str(quantized_path))
            optimization_results["dynamic_quantization"] = dynamic_results
            
            # FP16 conversion
            print("🔢 Converting to FP16...")
            fp16_path = self.deployment_dir / "model_fp16.pth"
            fp16_results = quantizer.convert_to_fp16(model_path, str(fp16_path))
            optimization_results["fp16_conversion"] = fp16_results
            
            # ONNX conversion
            print("🔄 Converting to ONNX...")
            onnx_path = self.deployment_dir / "model.onnx"
            onnx_results = onnx_converter.convert_to_onnx(model_path, str(onnx_path))
            optimization_results["onnx_conversion"] = onnx_results
            
            # Performance benchmarking
            print("⚡ Running performance benchmarks...")
            model_paths = {
                "pytorch": model_path,
                "onnx": str(onnx_path)
            }
            benchmark_results = benchmark.compare_models(model_paths)
            optimization_results["benchmark"] = benchmark_results
            
            # Create benchmark visualization
            benchmark.plot_benchmark_results(benchmark_results)
            
            print("✅ Model optimization completed!")
            self.deployment_results["optimization"] = optimization_results
            
            return optimization_results
            
        except Exception as e:
            print(f"❌ Model optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def build_docker_image(self, tag: str = "medical-vit", target: str = "production") -> Dict[str, Any]:
        """Build Docker image"""
        print("🐳 Building Docker image...")
        
        try:
            # Build command
            cmd = [
                "docker", "build",
                "-t", tag,
                "--target", target,
                "."
            ]
            
            # Run build
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Docker image built successfully!")
                return {
                    "status": "success",
                    "tag": tag,
                    "target": target,
                    "output": result.stdout
                }
            else:
                print(f"❌ Docker build failed: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr
                }
                
        except Exception as e:
            print(f"❌ Docker build failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_docker_container(self, image_tag: str = "medical-vit", port: int = 8000) -> Dict[str, Any]:
        """Run Docker container"""
        print("🚀 Starting Docker container...")
        
        try:
            # Stop existing container if running
            subprocess.run(["docker", "stop", "medical-vit-api"], capture_output=True)
            subprocess.run(["docker", "rm", "medical-vit-api"], capture_output=True)
            
            # Run container
            cmd = [
                "docker", "run",
                "-d",
                "--name", "medical-vit-api",
                "-p", f"{port}:8000",
                "-v", f"{Path.cwd()}/models:/app/models:ro",
                "-v", f"{Path.cwd()}/configs:/app/configs:ro",
                image_tag
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                container_id = result.stdout.strip()
                print(f"✅ Container started: {container_id}")
                
                # Wait for container to be ready
                print("⏳ Waiting for API to be ready...")
                time.sleep(10)
                
                # Test API
                health_check = self._test_api_health(f"http://localhost:{port}")
                
                return {
                    "status": "success",
                    "container_id": container_id,
                    "port": port,
                    "health_check": health_check
                }
            else:
                print(f"❌ Container start failed: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr
                }
                
        except Exception as e:
            print(f"❌ Container start failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _test_api_health(self, api_url: str) -> Dict[str, Any]:
        """Test API health"""
        try:
            import requests
            response = requests.get(f"{api_url}/health", timeout=10)
            if response.status_code == 200:
                return {"status": "healthy", "response": response.json()}
            else:
                return {"status": "unhealthy", "status_code": response.status_code}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def deploy_with_docker_compose(self, profile: str = "production") -> Dict[str, Any]:
        """Deploy using Docker Compose"""
        print("🐳 Deploying with Docker Compose...")
        
        try:
            # Stop existing services
            subprocess.run(["docker-compose", "down"], capture_output=True)
            
            # Start services
            cmd = ["docker-compose", "up", "-d"]
            if profile != "production":
                cmd.extend(["--profile", profile])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Docker Compose deployment successful!")
                
                # Wait for services to be ready
                time.sleep(15)
                
                # Test API
                health_check = self._test_api_health("http://localhost:8000")
                
                return {
                    "status": "success",
                    "profile": profile,
                    "health_check": health_check,
                    "output": result.stdout
                }
            else:
                print(f"❌ Docker Compose deployment failed: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr
                }
                
        except Exception as e:
            print(f"❌ Docker Compose deployment failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_performance_test(self, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Run performance tests on deployed API"""
        print("⚡ Running performance tests...")
        
        try:
            from api_client import MedicalViTClient, APITester
            
            client = MedicalViTClient(api_url)
            tester = APITester(client)
            
            # Run all tests
            test_results = tester.run_all_tests()
            
            print("✅ Performance tests completed!")
            return {
                "status": "success",
                "test_results": test_results
            }
            
        except Exception as e:
            print(f"❌ Performance tests failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        print("📊 Generating deployment report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deployment_results": self.deployment_results,
            "summary": {
                "optimization_successful": "optimization" in self.deployment_results,
                "containerization_successful": "containerization" in self.deployment_results,
                "api_deployment_successful": "api_deployment" in self.deployment_results,
                "performance_tested": "performance" in self.deployment_results
            }
        }
        
        # Save report
        report_path = self.deployment_dir / "deployment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📁 Deployment report saved to: {report_path}")
        return report
    
    def full_deployment(self, model_path: str = "models/best_model.pth") -> Dict[str, Any]:
        """Run complete deployment pipeline"""
        print("🚀 Starting full deployment pipeline...")
        
        deployment_steps = [
            ("Model Optimization", lambda: self.optimize_model(model_path)),
            ("Docker Build", lambda: self.build_docker_image()),
            ("Docker Compose Deploy", lambda: self.deploy_with_docker_compose()),
            ("Performance Test", lambda: self.run_performance_test())
        ]
        
        results = {}
        for step_name, step_func in deployment_steps:
            print(f"\n📋 {step_name}...")
            try:
                result = step_func()
                results[step_name] = result
                if result.get("status") == "failed":
                    print(f"❌ {step_name} failed, stopping deployment")
                    break
                print(f"✅ {step_name} completed")
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                results[step_name] = {"status": "failed", "error": str(e)}
                break
        
        # Generate final report
        final_report = self.generate_deployment_report()
        
        print("\n🎉 Deployment pipeline completed!")
        return {
            "deployment_results": results,
            "final_report": final_report
        }


def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(description="Deploy Medical Vision Transformer")
    parser.add_argument("--model-path", default="models/best_model.pth", help="Path to model file")
    parser.add_argument("--optimize-only", action="store_true", help="Only optimize model")
    parser.add_argument("--build-only", action="store_true", help="Only build Docker image")
    parser.add_argument("--deploy-only", action="store_true", help="Only deploy with Docker Compose")
    parser.add_argument("--test-only", action="store_true", help="Only run performance tests")
    parser.add_argument("--full", action="store_true", help="Run full deployment pipeline")
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    deployer = DeploymentManager()
    
    if args.optimize_only:
        deployer.optimize_model(args.model_path)
    elif args.build_only:
        deployer.build_docker_image()
    elif args.deploy_only:
        deployer.deploy_with_docker_compose()
    elif args.test_only:
        deployer.run_performance_test()
    elif args.full:
        deployer.full_deployment(args.model_path)
    else:
        print("Please specify a deployment option. Use --help for more information.")


if __name__ == "__main__":
    main()
