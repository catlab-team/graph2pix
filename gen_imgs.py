import os
import cv2
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

from PIL import Image
from tqdm import tqdm

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

BASE_DIR = f"gen_images/{opt.name}/"
GT_DIR = f"{BASE_DIR}gts"
PRED_DIR = f"{BASE_DIR}generated"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
print(PRED_DIR)

for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        
    data_label = torch.stack(data['label']) if opt.mv else data['label']
    if opt.data_type == 16:
        data_label = data_label.half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data_label = data_label.uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:        
#         print(data_label.size())
        generated = model.inference(data_label, data['inst'], data['image'])
        
    if opt.mv:
        data_label = data['label'][0]
        img_path = data['path'][0]
        gt_im = img_path[0].replace('parent_0_0', 'z')
    else:
        data_label = data['label']
        img_path = data['path']
        gt_im = img_path[0].replace('parent_0_0', 'z')

    im = cv2.cvtColor(util.tensor2im(generated.data[0]), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{PRED_DIR}/{gt_im.split('/')[-1]}", im)
    
    Image.open(gt_im).resize((opt.loadSize, opt.loadSize)).save(os.path.join(GT_DIR, gt_im.split('/')[-1]))


