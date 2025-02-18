# Neural Child Troubleshooting Guide

## Common Errors and Solutions

### 1. Connection Errors
```
HTTPConnectionPool(host='0.0.0.0', port=1234): Max retries exceeded
```
**Cause**: The LLM or embedding server is not running or not accessible.
**Solution**: 
- Ensure the local API server is running
- Change server address from `0.0.0.0` to `localhost` in configuration
- Update environment variables:
  ```python
  os.environ['EMBEDDING_SERVER_HOST'] = 'localhost'
  os.environ['CHAT_SERVER_HOST'] = 'localhost'
  ```
- Verify network connectivity

### 2. Shape Mismatch Errors
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x4 and 128x256)
```
**Cause**: Emotional input tensor (1x4) not properly projected to model's input size.
**Solution**:
- Use emotion projection layer before network operations:
  ```python
  if input_tensor.size(1) == 4:  # If emotional input
      input_tensor = brain.emotion_projection_layer(input_tensor)
  ```
- Verify tensor shapes at each step with debug prints:
  ```python
  print(f"Input tensor shape before projection: {input_tensor.size()}")
  print(f"Input tensor shape after projection: {input_tensor.size()}")
  ```
- Ensure emotion projection layer is properly initialized:
  ```python
  self.emotion_projection_layer = nn.Sequential(
      nn.Linear(4, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, input_size)
  )
  ```

### 3. Missing Attribute Errors
```
'DynamicNeuralChild' object has no attribute 'get_emotional_state'
```
**Cause**: Required method not implemented in class.
**Solution**:
- Implement missing methods in relevant classes
- Check for typos in method names
- Verify class inheritance and implementation

### 4. Index Errors
```
IndexError: list index out of range
```
**Cause**: Attempting to access non-existent list elements.
**Solution**:
- Add error handling for empty lists
- Provide default values
- Check list length before accessing elements

### 5. CUDA/GPU Errors
```
AssertionError: Torch not compiled with CUDA enabled
```
**Cause**: CUDA support missing or GPU not available.
**Solution**:
- Install CUDA-enabled PyTorch version
- Check GPU availability with `torch.cuda.is_available()`
- Fallback to CPU if needed

## Prevention Tips

1. **Input Validation**
   - Always validate input dimensions
   - Provide default values
   - Add error handling for edge cases

2. **Server Connectivity**
   - Implement retry mechanisms
   - Add timeout handling
   - Provide fallback responses

3. **Resource Management**
   - Monitor GPU memory usage
   - Implement garbage collection
   - Use context managers for resources

4. **Error Logging**
   - Implement comprehensive error logging
   - Include stack traces
   - Log relevant state information

## Quick Fixes

### Server Issues
```bash
# Check if server is running
curl http://localhost:1234/health

# Restart server
sudo systemctl restart llm-server
```

### CUDA Issues
```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
```

### Memory Issues
```python
# Clear CUDA cache
torch.cuda.empty_cache()

# Check memory usage
print(f"Memory allocated: {torch.cuda.memory_allocated()}")
```

## Getting Help

1. Check the logs in `logs/` directory
2. Review relevant documentation
3. Search issue tracker for similar problems
4. Contact support team with:
   - Error message
   - System configuration
   - Steps to reproduce
   - Relevant logs 