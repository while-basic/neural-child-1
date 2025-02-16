import torch
import os

def register_custom_classes():
    """Initialize and register custom torch classes"""
    try:
        # Clear existing registrations
        torch._C._jit_clear_class_registry()
        
        # Ensure CUDA is available if needed
        if torch.cuda.is_available():
            torch.cuda.init()
            
        # Force CPU execution for custom classes
        os.environ['PYTORCH_JIT'] = '1'
        torch._C._set_torch_dispatch_mode('python')
        
        return True
    except Exception as e:
        print(f"Error registering custom classes: {str(e)}")
        return False
