import pynvml

pynvml.nvmlInit()
gpu_name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)).decode('UTF-8')

if('K80' not in gpu_name):
  print('***********************************************************************')
  print('Woo! Your instance has the right kind of GPU, a '+ str(gpu_name)+'!')
  print('***********************************************************************')
  print()
else:
  raise Exception("""
                  Unfortunately Colab didn't give you a RAPIDS compatible GPU (P4, P100, T4, or V100), but gave you a """+ gpu_name +""".
  
                  Make sure you've configured Colab to request a GPU Instance Type.
                  
                  If you get an incompatible GPU (i.e., a K80), use 'Runtime -> Factory Reset Runtimes...' to try again"""
                  )