from inference import SegRetino

# Initializing the SegRetino Inference
seg = SegRetino(<path/to/test/img.png>)

# Running inference
seg.inference(set_weight_dir = 'unet.pth', path = 'output.png', blend_path = 'blend.png')
