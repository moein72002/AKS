import torch

if torch.cuda.is_available():
    print("✅ CUDA (GPU) is available!")
    
    # Get the number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s).")
    
    # Loop through each GPU and print its name
    for i in range(gpu_count):
        print(f"--- GPU {i}: {torch.cuda.get_device_name(i)}")
        
    # Get the currently selected device
    current_device = torch.cuda.current_device()
    print(f"Current GPU index: {current_device}")
    print(f"Current GPU name: {torch.cuda.get_device_name(current_device)}")

else:
    print("❌ No CUDA (GPU) found. PyTorch will use the CPU.")