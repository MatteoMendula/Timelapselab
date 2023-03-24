

def get_framework_extension(framework: str = 'torch'):
    extension_dict = {'pytorch': 'pt',
                      'torch': 'pt',
                      'torchscript': 'torchscript',
                      'onnx': 'onnx',
                      'openvino': 'xml',
                      'coreml': 'mlmodel',
                      'tensorrt': 'engine',
                      'tensorflow': 'pb',
                      'tflite': 'tflite',
                      'edgetpu': 'tflite',}
    print('\n{}\n'.format(framework.lower()))
    return extension_dict[framework.lower()]
