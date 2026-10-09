import importlib.util,json,sys,tempfile,pathlib,torch,numpy as np,cv2
root=pathlib.Path(sys.argv[1]);base=pathlib.Path(sys.argv[2]);torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available()
spec=importlib.util.spec_from_file_location('subject',root/'scripts/model_convert.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
image=cv2.imread(str(root/'assets/bus.jpg'));h,w=image.shape[:2];ratio=min(640/h,640/w);nw,nh=round(w*ratio),round(h*ratio)
canvas=np.full((640,640,3),114,dtype=np.uint8);canvas[(640-nh)//2:(640-nh)//2+nh,(640-nw)//2:(640-nw)//2+nw]=cv2.resize(image,(nw,nh))
bus=torch.from_numpy(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float().div(255).unsqueeze(0)
np.save(base/'bus-input.npy',bus.numpy())
rows=[]
for stem in ('yolov8n','yolov5n'):
 source=root/'models'/(stem+'.pt');artifact=base/(stem+'-640.torchscript')
 original=m.load_model(source,torch)[0]
 script=torch.jit.load(str(artifact),map_location='cpu').eval()
 for tag,input in [('bus',bus),('gradient',torch.linspace(0,1,3*640*640).reshape(1,3,640,640))]:
  with torch.inference_mode():eager=m.primary_output(original(input),torch);cpu=script(input)
  np.save(base/(stem+'-'+tag+'-expected.npy'),eager.numpy())
  assert torch.allclose(cpu,eager,rtol=.002,atol=.002),('CPU differs',stem,tag,float((cpu-eager).abs().max()))
  gpu=torch.jit.load(str(artifact),map_location='cpu').to('cuda').eval()
  with torch.inference_mode():actual=gpu(input.to('cuda')).cpu()
  original.to('cuda')
  with torch.inference_mode(): eager_cuda=m.primary_output(original(input.to('cuda')),torch).cpu()
  difference=(actual-eager).abs();index=np.unravel_index(int(difference.argmax()),tuple(difference.shape))
  print(json.dumps({'raw_cpu_cuda_worst_index':list(map(int,index)),'cpu_value':float(eager[index]),'cuda_value':float(actual[index]),'cuda_original_cuda_max_abs':float((actual-eager_cuda).abs().max()),'cuda_original_cuda_allclose':bool(torch.allclose(actual,eager_cuda,rtol=.002,atol=.002))}),flush=True)
  assert torch.allclose(actual,eager_cuda,rtol=.002,atol=.002),('CUDA selected eager differs',stem,tag,float((actual-eager_cuda).abs().max()))
  print(json.dumps({'cuda_diff_scores':float((actual[:,4:]-eager_cuda[:,4:]).abs().max()) if stem=='yolov8n' else float((actual[:,:,4:]-eager_cuda[:,:,4:]).abs().max())}),flush=True)
  unoptimized=torch.jit.load(str(artifact),map_location='cpu').to('cuda').eval()
  with torch.inference_mode(),torch.jit.optimized_execution(False): diagnostic=unoptimized(input.to('cuda')).cpu()
  print(json.dumps({'unoptimized_cuda_eager_max_abs':float((diagnostic-eager_cuda).abs().max()),'unoptimized_allclose':bool(torch.allclose(diagnostic,eager_cuda,rtol=.002,atol=.002))}),flush=True)
  original.to('cpu')
  row={'cuda_eager_max_abs':float((actual-eager_cuda).abs().max()),'model':stem,'input':tag,'shape':list(cpu.shape),'cpu_original_max_abs':float((cpu-eager).abs().max()),'cuda_original_max_abs':float((actual-eager).abs().max()),'buffer_names':[name for name,_ in gpu.named_buffers() if '_vision_static_' in name],'cuda_buffers':all(value.device.type=='cuda' for _,value in gpu.named_buffers())}
  assert row['cuda_buffers'];rows.append(row);print(json.dumps(row),flush=True)
  del gpu;torch.cuda.empty_cache()
 (base/(stem+'-smoke-metadata.json')).write_text(json.dumps({'conversion':json.loads((base/(stem+'-onnx.jsonl')).read_text().splitlines()[-1])}))
(base/'torchscript-cpu-cuda-result.json').write_text(json.dumps({'success':True,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'comparisons':rows},indent=2))
