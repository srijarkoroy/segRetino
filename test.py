from inference import SegRetino

# Initializing the SegRetino Inference
seg = SegRetino(<path/to/test/img.png>)
seg.inference(set_weight_dir = 'unet.pth', set_gen_dir = 'result_img')
