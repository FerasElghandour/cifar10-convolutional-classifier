"""Predict a local RGB image with a trained released architecture."""
from pathlib import Path
import argparse,json,numpy as np,torch
from PIL import Image
from train import build_model
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('image',type=Path);a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);config=ck['config'];model=build_model(config);model.load_state_dict(ck['model_state'],strict=True);model.eval()
with Image.open(a.image) as image:x=np.asarray(image.convert('RGB').resize((32,32))).copy().transpose(2,0,1)
with torch.no_grad():probs=model(torch.from_numpy(x).unsqueeze(0).float()/127.5-1).softmax(-1)[0].tolist()
print(json.dumps({'class_index':int(np.argmax(probs)),'class_names':config.get('class_names'),'probabilities':probs}))
