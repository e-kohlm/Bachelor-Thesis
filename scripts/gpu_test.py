from pynvml import *
from pynvml.smi import nvidia_smi
import torch
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer#, logging
from datasets import load_dataset, load_from_disk


#logging.set_verbosity_error()

def print_gpu_utilization():
    nvmlInit()
    print("Driver Version:", nvmlSystemGetDriverVersion())
    deviceCount = nvmlDeviceGetCount()
    
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("Device", i, ":", nvmlDeviceGetName(handle))

    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)    
    print(f"  GPU memory occupied: {info.used//1024**2} MB.")
    allocated_memory = torch.cuda.memory_allocated()
    print(f"  Current GPU memory allocated: {allocated_memory / 1024**3:.2f} GB") 
    memory_reserved = torch.cuda.memory_reserved()
    print(f"  GPU memory reserved: {memory_reserved / 1024**3:.2f} GB")

    nvsmi = nvidia_smi.getInstance()
    device_query = nvsmi.DeviceQuery('memory.free, memory.total')
    print(f" Device query:{device_query}")
    
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")
        print(f"Memory Free: {torch.cuda.memory_stats(i)['reserved_bytes.all.current']/1024**3:.2f} GB") 
