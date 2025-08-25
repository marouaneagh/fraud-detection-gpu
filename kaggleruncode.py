# 🚀 ONE-COMMAND FRAUD DETECTION PIPELINE - MAROUANE'S REPO
import subprocess
import sys

print("🚀 MAROUANE'S GPU FRAUD DETECTION PIPELINE")
print("=" * 50)

# Install GPU requirements
print("📦 Installing GPU dependencies...")

# First, remove any conflicting CuPy installations
print("   🔄 Removing conflicting CuPy packages...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "cupy-cuda11x", "cupy-cuda12x", "cupy", "-y"], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Install the correct CuPy version for Kaggle's environment (CUDA 12.x)
print("   📥 Installing cupy-cuda12x...")
subprocess.run([sys.executable, "-m", "pip", "install", "cupy-cuda12x", "-q"])

# Install other GPU dependencies
print("   📥 Installing other GPU dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "optuna", "imbalanced-learn", "-q"])

# Download your specific repo
import requests
import zipfile
import os

print("📥 Downloading from marouaneagh/fraud-detection-gpu...")
url = "https://github.com/marouaneagh/fraud-detection-gpu/archive/refs/heads/main.zip"
response = requests.get(url)

with open("/kaggle/working/pipeline.zip", "wb") as f:
    f.write(response.content)

print("📂 Extracting files...")
# Extract
with zipfile.ZipFile("/kaggle/working/pipeline.zip", "r") as zip_ref:
    zip_ref.extractall("/kaggle/working/")

# Find extracted folder (GitHub adds hash)
extracted = [d for d in os.listdir("/kaggle/working/") if d.startswith("fraud-detection-gpu")]
repo_path = f"/kaggle/working/{extracted[0]}"

print(f"✅ Extracted to: {repo_path}")

# Add to path and change directory
sys.path.insert(0, repo_path)
os.chdir(repo_path)

print("🚀 Starting GPU-accelerated pipeline...")

# Check what files we have
print("📁 Available files:")
for file in os.listdir():
    if file.endswith('.py'):
        print(f"   {file}")

# Import and run the main pipeline
try:
    # Clear any existing command line arguments to avoid conflicts
    import sys
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep only the script name
    
    # Execute the main function directly to avoid argument issues
    from run_all_gpu import main
    success = main()
    
    if success:
        print("🎉 Pipeline completed successfully!")
    else:
        print("❌ Pipeline completed with errors")
        
    # Restore original arguments
    sys.argv = original_argv
    
except Exception as e:
    print(f"❌ Error: {e}")
    # Try individual steps if main runner fails
    print("🔄 Trying individual steps...")
    try:
        from data_loader import main as data_loader_main
        data_loader_main()
        print("✅ Step 1 completed")
    except Exception as e2:
        print(f"❌ Step 1 failed: {e2}")
