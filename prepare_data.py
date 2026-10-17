"""Prepare official CIFAR-10 splits from the uoft-cs parquet mirror.

Source: https://huggingface.co/datasets/uoft-cs/cifar10
Dataset creators: Alex Krizhevsky, Vinod Nair and Geoffrey Hinton.
The mirror SHA256 values are checked before image decoding. Test rows never enter
training/validation. Optional limits define a new reduced-data protocol.
"""
import argparse,hashlib,io,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.model_selection import train_test_split
HASHES={'train':'8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883','test':'841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de'}
def load(path,split):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=HASHES[split]:raise ValueError('Unexpected mirror bytes for '+split)
    table=pq.read_table(path).to_pydict()
    x=np.stack([np.asarray(Image.open(io.BytesIO(r['bytes'])).convert('RGB')).transpose(2,0,1) for r in table['img']])
    return x,np.array(table['label'],dtype=np.int64)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--train',type=Path,required=True);ap.add_argument('--test',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit-train',type=int);ap.add_argument('--limit-test',type=int);a=ap.parse_args()
    if (a.limit_train is not None and a.limit_train<20) or (a.limit_test is not None and a.limit_test<2):raise ValueError('Subset budgets require at least20 training and2 test rows')
    if a.output.exists():raise ValueError('Refusing to overwrite prepared data')
    x,y=load(a.train,'train');xt,yt=load(a.test,'test')
    tr,va=train_test_split(np.arange(len(y)),test_size=5000,stratify=y,random_state=7)
    if a.limit_train:
        if a.limit_train<20:raise ValueError('At least20 training samples')
        tr=tr[:a.limit_train];va=va[:max(20,a.limit_train//5)]
    ti=np.arange(len(yt))[:a.limit_test]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,x_train=x[tr],y_train=y[tr],x_val=x[va],y_val=y[va],x_test=xt[ti],y_test=yt[ti])
    a.output.with_suffix('.json').write_text(json.dumps({'source':'uoft-cs/cifar10 mirror of official CIFAR-10','source_sha256':HASHES,'seed':7,'training_indices':tr.tolist(),'validation_indices':va.tolist(),'test_indices':ti.tolist(),'normalization':'uint8 -> float32 /127.5 -1 in trainer','reduced_protocol':bool(a.limit_train or a.limit_test)},indent=2)+'\n')
if __name__=='__main__':main()
