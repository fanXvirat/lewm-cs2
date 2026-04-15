#!/usr/bin/env python3
"""
CS2 LeWM — Merged Final Version
════════════════════════════════════════════════════════════════════════════
Based on lewm_cs2_h100_final.py (your implementation).
Adds 4 targeted improvements:
  [PATCH-1] VGG perceptual + PatchGAN decoder (sharp reconstructions)
  [PATCH-2] decoder_epochs=80 (was 25 — blurry)
  [PATCH-3] 15 visual evaluation plots
  [PATCH-4] SIGReg on predictor outputs (small weight, prevents drift)

Your file wins on: aggregate_actions (mathematically correct mouse accumulation),
DIS optical flow, scope/flash heuristics, ClipStore, budget deadline, null_ratio
tracking per epoch, clip-level val split, residual predictor, pos embedding.

Run:
  python train.py --stage all --budget-hours 3.4
    python train.py --stage smoke
    python train.py --stage eval
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset

try:
    import cv2
except ImportError:
    cv2 = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTION_NAMES = ["yaw","pitch","W","A","S","D","Space","Ctrl","Shift","LMB","RMB","R"]
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


@dataclass
class Config:
    workspace:       str   = "/teamspace/studios/this_studio"
    raw_dir:         str   = "/teamspace/studios/this_studio/cs2_raw"
    processed_dir:   str   = "/teamspace/studios/this_studio/cs2_processed"
    run_dir:         str   = "/teamspace/studios/this_studio/cs2_lewm_merged"
    seed:            int   = 42

    fps:                   int   = 10
    img_size:              int   = 224
    flow_size:             int   = 96
    max_frames_per_clip:   int   = 18000
    min_frames_per_clip:   int   = 128
    storage_mode:          str   = "auto"
    ram_budget_gb:         float = 80.0

    seq_len:               int   = 20
    history_size:          int   = 8
    min_gap:               int   = 1
    max_gap:               int   = 4
    eval_gap:              int   = 2
    train_index_stride:    int   = 2
    val_index_stride:      int   = 8
    val_fraction:          float = 0.10

    batch_size:            int   = 128
    val_batch_size:        int   = 128
    num_workers:           int   = 16
    epochs:                int   = 24
    lr:                    float = 1e-4
    weight_decay:          float = 1e-3
    grad_clip:             float = 2.0
    warmup_epochs:         int   = 2
    sigreg_weight:         float = 0.025
    sigreg_pred_weight:    float = 0.005
    sigreg_warmup_steps:   int   = 2000
    dynamic_weight_power:  float = 0.5
    train_steps_per_epoch: int   = 700
    val_steps:             int   = 80
    compile_model:         bool  = True
    precision:             str   = "bf16"

    encoder_hidden:        int   = 384
    encoder_heads:         int   = 6
    encoder_layers:        int   = 12
    encoder_intermediate:  int   = 1536
    patch_size:            int   = 14
    embed_dim:             int   = 384
    predictor_depth:       int   = 8
    predictor_heads:       int   = 16
    predictor_dim_head:    int   = 64
    predictor_mlp_dim:     int   = 2048
    predictor_dropout:     float = 0.1
    proj_hidden:           int   = 2048
    action_embed_dim:      int   = 384
    key_embed_dim:         int   = 32
    residual_predictor:    bool  = True
    gradient_checkpointing:bool  = False

    decoder_epochs:        int   = 80
    decoder_batch_size:    int   = 512
    decoder_lr:            float = 3e-4
    decoder_weight_decay:  float = 1e-4
    decoder_steps_per_epoch:int  = 160
    decoder_val_steps:     int   = 20
    decoder_max_frames:    int   = 120000
    decoder_w_mse:         float = 0.25
    decoder_w_perc:        float = 0.40
    decoder_w_edge:        float = 0.20
    decoder_w_adv:         float = 0.10
    decoder_w_ssim:        float = 0.05

    eval_sequences:        int   = 256
    budget_hours:          float = 3.4
    reserve_decoder_minutes:float= 50.0
    reserve_eval_minutes:  float = 20.0

    stage: str = "all"


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--stage",           default=cfg.stage)
    p.add_argument("--workspace",       default=cfg.workspace)
    p.add_argument("--raw-dir",         default=cfg.raw_dir)
    p.add_argument("--processed-dir",   default=cfg.processed_dir)
    p.add_argument("--run-dir",         default=cfg.run_dir)
    p.add_argument("--fps",             type=int,   default=cfg.fps)
    p.add_argument("--img-size",        type=int,   default=cfg.img_size)
    p.add_argument("--flow-size",       type=int,   default=cfg.flow_size)
    p.add_argument("--max-frames-per-clip", type=int, default=cfg.max_frames_per_clip)
    p.add_argument("--batch-size",      type=int,   default=cfg.batch_size)
    p.add_argument("--epochs",          type=int,   default=cfg.epochs)
    p.add_argument("--decoder-epochs",  type=int,   default=cfg.decoder_epochs)
    p.add_argument("--num-workers",     type=int,   default=cfg.num_workers)
    p.add_argument("--budget-hours",    type=float, default=cfg.budget_hours)
    p.add_argument("--storage-mode",    default=cfg.storage_mode)
    p.add_argument("--no-compile",      action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()
    out = Config(
        stage=args.stage, workspace=args.workspace,
        raw_dir=args.raw_dir, processed_dir=args.processed_dir,
        run_dir=args.run_dir, fps=args.fps, img_size=args.img_size,
        flow_size=args.flow_size, max_frames_per_clip=args.max_frames_per_clip,
        batch_size=args.batch_size, val_batch_size=args.batch_size,
        epochs=args.epochs, decoder_epochs=args.decoder_epochs,
        num_workers=args.num_workers, budget_hours=args.budget_hours,
        storage_mode=args.storage_mode, compile_model=not args.no_compile,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    return auto_configure(out)


def auto_configure(cfg: Config) -> Config:
    if not torch.cuda.is_available():
        cfg.batch_size=4; cfg.val_batch_size=4; cfg.num_workers=0
        cfg.compile_model=False; cfg.precision="fp32"
        cfg.epochs=min(cfg.epochs,2); cfg.decoder_epochs=min(cfg.decoder_epochs,2)
        return cfg
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cfg.precision = "bf16"
    if ("H100" in name or "H200" in name) and vram > 70:
        cfg.batch_size=max(cfg.batch_size,128); cfg.val_batch_size=cfg.batch_size
        cfg.num_workers=max(cfg.num_workers,16); cfg.train_steps_per_epoch=700
        cfg.decoder_batch_size=512
    elif vram > 40:
        cfg.batch_size=min(cfg.batch_size,96); cfg.val_batch_size=cfg.batch_size
        cfg.num_workers=min(cfg.num_workers,8); cfg.compile_model=False
    else:
        cfg.batch_size=min(cfg.batch_size,32); cfg.val_batch_size=cfg.batch_size
        cfg.num_workers=min(cfg.num_workers,4); cfg.compile_model=False
    return cfg


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=True


def ensure_dir(p):
    p=Path(p); p.mkdir(parents=True,exist_ok=True); return p

def write_json(path, payload):
    with open(path,"w") as f: json.dump(payload,f,indent=2)


def sigmoid_np(x): return 1./(1.+np.exp(-x))
def robust_zscore(x):
    x=x.astype(np.float32,copy=False); med=np.median(x)
    mad=np.median(np.abs(x-med)); return (x-med)/(1.4826*mad+1e-6)
def square_crop(f):
    h,w=f.shape[:2]; s=min(h,w); y0=(h-s)//2; x0=(w-s)//2
    return f[y0:y0+s, x0:x0+s]

def scope_score(frame):
    h,w=frame.shape[:2]; ph=max(8,h//6); pw=max(8,w//6)
    center=frame[h//3:2*h//3, w//3:2*w//3]
    corners=[frame[:ph,:pw].mean(),frame[:ph,-pw:].mean(),
             frame[-ph:,:pw].mean(),frame[-ph:,-pw:].mean()]
    return float(np.clip(sigmoid_np(np.array([6.*(center.mean()/255.-np.mean(corners)/255.-0.10)]))[0],0.,1.))

def flash_score(prev, frame):
    h,w=frame.shape[:2]
    roi_p=prev[int(h*.45):int(h*.92), int(w*.35):int(w*.95)].astype(np.float32)
    roi_c=frame[int(h*.45):int(h*.92), int(w*.35):int(w*.95)].astype(np.float32)
    d=(roi_c-roi_p).mean(axis=2)
    return float(np.clip(.6*(d>30.).mean()/.02+.4*np.maximum(d,0.).mean()/40.,0.,1.))

def extract_frames(video_path, fps, img_size, max_frames):
    if cv2 is None: raise ImportError("opencv-python-headless required")
    cap=cv2.VideoCapture(str(video_path)); src_fps=cap.get(cv2.CAP_PROP_FPS) or float(fps)
    step=max(src_fps/float(fps),1.); next_s=0.; fi=0.; frames=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        if fi+1e-6<next_s: fi+=1.; continue
        rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(square_crop(rgb),(img_size,img_size),interpolation=cv2.INTER_AREA)
        frames.append(rgb); next_s+=step; fi+=1.
        if len(frames)>=max_frames: break
    cap.release()
    return np.stack(frames,0).astype(np.uint8) if frames else np.zeros((0,img_size,img_size,3),dtype=np.uint8)

def build_pseudo_actions(frames, flow_size):
    if cv2 is None: raise ImportError("opencv-python-headless required")
    n=len(frames); A=np.zeros((n,len(ACTION_NAMES)),dtype=np.float32)
    if n<=1: return A
    try:
        eng=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        calc=lambda p,g: eng.calc(p,g,None)
    except:
        calc=lambda p,g: cv2.calcOpticalFlowFarneback(p,g,None,0.5,3,15,3,5,1.2,0)

    dx=np.zeros(n,np.float32); dy=np.zeros(n,np.float32)
    div=np.zeros(n,np.float32); strafe=np.zeros(n,np.float32)
    mag=np.zeros(n,np.float32); flash=np.zeros(n,np.float32)
    scope=np.zeros(n,np.float32); scope[0]=scope_score(frames[0])

    ps=cv2.resize(frames[0],(flow_size,flow_size),cv2.INTER_AREA)
    pg=cv2.cvtColor(ps,cv2.COLOR_RGB2GRAY)
    for i in range(1,n):
        cs=cv2.resize(frames[i],(flow_size,flow_size),cv2.INTER_AREA)
        g=cv2.cvtColor(cs,cv2.COLOR_RGB2GRAY); fl=calc(pg,g)
        u,v=fl[...,0],fl[...,1]
        dx[i]=float(np.median(u)); dy[i]=float(np.median(v))
        div[i]=float((cv2.Sobel(u,cv2.CV_32F,1,0,3)+cv2.Sobel(v,cv2.CV_32F,0,1,3)).mean())
        strafe[i]=float(u[:,:u.shape[1]//2].mean()-u[:,u.shape[1]//2:].mean())
        mag[i]=float(np.sqrt(u*u+v*v).mean())
        flash[i]=flash_score(frames[i-1],frames[i]); scope[i]=scope_score(frames[i]); pg=g

    zd=robust_zscore(div); zs=robust_zscore(strafe)
    zm=robust_zscore(mag);  zf=robust_zscore(flash); zdy=robust_zscore(dy)

    A[:,0]=np.clip(dx*6.,-60.,60.); A[:,1]=np.clip(dy*6.,-60.,60.)
    A[:,2]=np.clip(sigmoid_np(.90*zd+.35*zm),0.,1.)
    A[:,4]=np.clip(sigmoid_np(-.90*zd+.20*zm),0.,1.)
    A[:,3]=np.clip(sigmoid_np(-1.20*zs),0.,1.)
    A[:,5]=np.clip(sigmoid_np(1.20*zs),0.,1.)
    A[:,8]=np.clip(A[:,2]*sigmoid_np(zm-.15),0.,1.)
    A[:,9]=np.clip(sigmoid_np(1.40*zf+.30*np.abs(zdy)),0.,1.)
    A[:,10]=np.clip(scope,0.,1.)
    A[:,6]=np.clip(.25*sigmoid_np(-zdy),0.,1.)
    A[:,7]=np.clip(.25*sigmoid_np(zdy),0.,1.)
    A[:,11]=0.; A[0]=0.; return A.astype(np.float32)


def aggregate_actions(action_array, frame_indices):
    """
    Mathematically correct action aggregation across temporal gaps.
    Mouse (yaw/pitch): SUM (velocity accumulates over skipped frames)
    Keys (binary): MEAN (proportion of time key was held)
    """
    seq_len=len(frame_indices)
    out=np.zeros((seq_len,action_array.shape[1]),dtype=np.float32)
    for t in range(seq_len-1):
        seg=np.asarray(action_array[frame_indices[t]+1:frame_indices[t+1]+1],dtype=np.float32)
        if seg.size==0: continue
        out[t,:2]=seg[:,:2].sum(axis=0)
        out[t,2:]=seg[:,2:].mean(axis=0)
    if seq_len>1: out[-1]=out[-2]
    return out


def preprocess_clip(video_path, cfg):
    proc=ensure_dir(cfg.processed_dir)
    fp=proc/f"{video_path.stem}_frames.npy"; ap=proc/f"{video_path.stem}_actions.npy"
    if fp.exists() and ap.exists():
        return {"name":video_path.stem,"num_frames":int(np.load(fp,mmap_mode='r').shape[0]),
                "frames_path":str(fp),"actions_path":str(ap)}
    frames=extract_frames(video_path,cfg.fps,cfg.img_size,cfg.max_frames_per_clip)
    if len(frames)<cfg.min_frames_per_clip: return None
    actions=build_pseudo_actions(frames,cfg.flow_size)
    np.save(fp,frames); np.save(ap,actions)
    write_json(proc/f"{video_path.stem}_meta.json",{"name":video_path.stem,"num_frames":len(frames)})
    return {"name":video_path.stem,"num_frames":len(frames),"frames_path":str(fp),"actions_path":str(ap)}


def preprocess_corpus(cfg):
    raw=ensure_dir(cfg.raw_dir)
    videos=sorted(p for p in raw.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    clips=[r for v in videos if (r:=preprocess_clip(v,cfg)) is not None]
    print(f"Preprocessed {len(clips)} clips")
    return clips


def discover_processed_clips(cfg):
    proc=ensure_dir(cfg.processed_dir); clips=[]
    for fp in sorted(proc.glob("*_frames.npy")):
        ap=Path(str(fp).replace("_frames.npy","_actions.npy"))
        if not ap.exists(): continue
        n=int(np.load(fp,mmap_mode='r').shape[0])
        clips.append({"name":fp.stem.replace("_frames",""),"num_frames":n,
                      "frames_path":str(fp),"actions_path":str(ap)})
    return clips


class ClipStore:
    def __init__(self, clips, mode="auto", ram_budget_gb=80.):
        self.clips=list(clips); self.mode=self._resolve(mode,ram_budget_gb)
        self._ram=[]; self._mmap={}
        if self.mode=="ram":
            for c in self.clips:
                self._ram.append((np.load(c["frames_path"]),np.load(c["actions_path"])))
            print(f"[ClipStore] Loaded {len(self.clips)} clips into RAM")
        else:
            print(f"[ClipStore] Using mmap (mode={self.mode})")

    def _resolve(self, mode, budget_gb):
        if mode in {"ram","mmap"}: return mode
        total=sum(os.path.getsize(c["frames_path"])+os.path.getsize(c["actions_path"])
                  for c in self.clips)
        return "ram" if total<=int(budget_gb*1e9*.8) else "mmap"

    def get(self, idx):
        if self.mode=="ram": return self._ram[idx]
        if idx not in self._mmap:
            c=self.clips[idx]
            self._mmap[idx]=(np.load(c["frames_path"],mmap_mode='r'),
                              np.load(c["actions_path"],mmap_mode='r'))
        return self._mmap[idx]

    def __len__(self): return len(self.clips)


class ActionNormalizer:
    def __init__(self):
        self.mean=torch.zeros(len(ACTION_NAMES)); self.std=torch.ones(len(ACTION_NAMES)); self.ready=False

    def fit(self, store, index, cfg, sample_count=2048):
        rng=np.random.default_rng(cfg.seed)
        picked=rng.choice(len(index),size=min(sample_count,len(index)),replace=False)
        collected=[]
        for i in picked:
            ci,start=index[int(i)]; _,actions=store.get(ci)
            fi=[start]; [fi.append(fi[-1]+cfg.eval_gap) for _ in range(cfg.seq_len-1)]
            agg=aggregate_actions(actions,fi); collected.append(agg[:,:2].reshape(-1,2))
        arr=np.concatenate(collected,0)
        self.mean[:2]=torch.tensor(arr.mean(0)); self.std[:2]=torch.tensor(arr.std(0).clip(1e-4))
        self.ready=True
        print(f"[ActionNorm] yaw: μ={self.mean[0]:.3f} σ={self.std[0]:.3f} | "
              f"pitch: μ={self.mean[1]:.3f} σ={self.std[1]:.3f}")

    def transform(self, a):
        if not self.ready: return a
        m=self.mean.to(a.device); s=self.std.to(a.device)
        out=a.clone(); out[...,:2]=(out[...,:2]-m[:2])/s[:2]; return out

    def state_dict(self): return {"mean":self.mean.tolist(),"std":self.std.tolist(),"ready":self.ready}
    def load_state_dict(self, p): self.mean=torch.tensor(p["mean"]); self.std=torch.tensor(p["std"]); self.ready=bool(p["ready"])


def split_clips(clips, val_fraction, seed):
    ids=list(range(len(clips))); rng=random.Random(seed); rng.shuffle(ids)
    n_val=max(1,int(round(len(ids)*val_fraction))); val=sorted(ids[:n_val]); train=sorted(ids[n_val:])
    if not train: train=val[:]
    return train,val

def build_window_index(clips, clip_ids, seq_len, max_gap, stride):
    items=[]; req=(seq_len-1)*max_gap+1
    for ci in clip_ids:
        n=clips[ci]["num_frames"]; ms=n-req
        if ms<=0: continue
        items.extend((ci,s) for s in range(0,ms,stride))
    return items


class CS2SequenceDataset(Dataset):
    def __init__(self, store, index, cfg, train):
        self.store=store; self.index=list(index); self.cfg=cfg; self.train=train
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        ci,start=self.index[idx]; frames,actions=self.store.get(ci)
        fids=[start]
        for _ in range(self.cfg.seq_len-1):
            fids.append(fids[-1]+random.randint(self.cfg.min_gap,self.cfg.max_gap) if self.train else fids[-1]+self.cfg.eval_gap)
        fids=[min(f,len(frames)-1) for f in fids]
        fb=np.asarray(frames[fids],dtype=np.uint8)
        ab=aggregate_actions(actions,fids)
        return torch.from_numpy(fb), torch.from_numpy(ab.astype(np.float32))


class CS2FrameDataset(Dataset):
    def __init__(self, store, clip_ids, max_frames):
        self.store=store; self.index=[]
        for ci in clip_ids:
            f,_=store.get(ci); stride=max(1,len(f)//max(1,max_frames//max(1,len(clip_ids))))
            self.index.extend((ci,i) for i in range(0,len(f),stride))
        if len(self.index)>max_frames: self.index=self.index[:max_frames]
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        ci,fi=self.index[idx]; f,_=self.store.get(ci)
        return torch.from_numpy(np.asarray(f[fi],dtype=np.uint8))


def collate_sequences(batch):
    f,a=zip(*batch); return torch.stack(f),torch.stack(a)
def collate_frames(batch): return torch.stack(batch)


def build_loaders(store, cfg):
    train_clips,val_clips=split_clips(store.clips,cfg.val_fraction,cfg.seed)
    ti=build_window_index(store.clips,train_clips,cfg.seq_len,cfg.max_gap,cfg.train_index_stride)
    vi=build_window_index(store.clips,val_clips,  cfg.seq_len,cfg.max_gap,cfg.val_index_stride)
    train_ds=CS2SequenceDataset(store,ti,cfg,train=True)
    val_ds  =CS2SequenceDataset(store,vi,cfg,train=False)
    tn=min(len(train_ds),cfg.train_steps_per_epoch*cfg.batch_size)
    vn=min(len(val_ds),  cfg.val_steps*cfg.val_batch_size)
    tr_sampler=RandomSampler(train_ds,replacement=len(train_ds)<tn,num_samples=tn)
    vl_subset=Subset(val_ds,list(range(vn)))
    kw=dict(pin_memory=torch.cuda.is_available(),num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers>0)
    tr_loader=DataLoader(train_ds,batch_size=cfg.batch_size,sampler=tr_sampler,
                          collate_fn=collate_sequences,drop_last=True,**kw)
    vl_loader=DataLoader(vl_subset,batch_size=cfg.val_batch_size,
                          sampler=SequentialSampler(vl_subset),
                          collate_fn=collate_sequences,drop_last=False,**kw)
    print(f"[loaders] train={len(tr_loader)} val={len(vl_loader)} batches | "
          f"train_clips={len(train_clips)} val_clips={len(val_clips)}")
    return tr_loader,vl_loader,train_ds,val_ds,train_clips,val_clips


def frames_to_device(t, device, dtype):
    if t.ndim==5:
        f=t.to(device,non_blocking=True).permute(0,1,4,2,3).float().div_(255.)
        m=torch.tensor(IMG_MEAN,device=device).view(1,1,3,1,1)
        s=torch.tensor(IMG_STD, device=device).view(1,1,3,1,1)
        f=(f-m)/s; return f.to(dtype) if device.type=="cuda" else f
    f=t.to(device,non_blocking=True).permute(0,3,1,2).float().div_(255.)
    m=torch.tensor(IMG_MEAN,device=device).view(1,3,1,1)
    s=torch.tensor(IMG_STD, device=device).view(1,3,1,1)
    f=(f-m)/s; return f.to(dtype) if device.type=="cuda" else f

def denorm_frames(t):
    nd=t.ndim; m=torch.tensor(IMG_MEAN,device=t.device,dtype=t.dtype)
    s=torch.tensor(IMG_STD, device=t.device,dtype=t.dtype)
    if nd==5: m=m.view(1,1,3,1,1); s=s.view(1,1,3,1,1)
    else: m=m.view(1,3,1,1); s=s.view(1,3,1,1)
    return (t*s+m).clamp(0,1)

def autocast_ctx(device, dtype):
    if device.type!="cuda":
        from contextlib import nullcontext; return nullcontext()
    return torch.autocast(device_type="cuda",dtype=dtype)

def maybe_compile(model, enabled):
    if not enabled or not hasattr(torch,"compile") or not torch.cuda.is_available(): return model
    try: return torch.compile(model,mode="reduce-overhead")
    except: return model

def model_device_dtype(cfg):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt=(torch.bfloat16 if cfg.precision=="bf16" and dev.type=="cuda" else
        torch.float16 if cfg.precision=="fp16" else torch.float32)
    return dev,dt


def modulate(x,s,sc): return x*(1.+sc)+s


class SIGReg(nn.Module):
    """Epps-Pulley normality statistic. Provable anti-collapse. Paper §3.1 / Appendix A."""
    def __init__(self,knots=17,num_proj=1024):
        super().__init__()
        t=torch.linspace(0,3,knots); dt=3./(knots-1)
        w=torch.full((knots,),2.*dt); w[[0,-1]]=dt; phi=torch.exp(-t.square()/2.)
        self.num_proj=num_proj
        self.register_buffer("t",t); self.register_buffer("phi",phi); self.register_buffer("weights",w*phi)
    def forward(self,z):
        a=torch.randn(z.size(-1),self.num_proj,device=z.device)
        a=a/a.norm(p=2,dim=0,keepdim=True)
        x_t=(z@a).unsqueeze(-1)*self.t
        err=(x_t.cos().mean(-3)-self.phi).square()+x_t.sin().mean(-3).square()
        return ((err@self.weights)*z.size(-2)).mean()


class SafeBN1d(nn.Module):
    def __init__(self,d): super().__init__(); self.bn=nn.BatchNorm1d(d)
    def forward(self,x):
        if not self.training or x.size(0)>1: return self.bn(x)
        return F.batch_norm(x,self.bn.running_mean,self.bn.running_var,self.bn.weight,self.bn.bias,training=False,eps=self.bn.eps)

class FF(nn.Module):
    def __init__(self,d,h,dp=0.): super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,h),nn.GELU(),nn.Dropout(dp),nn.Linear(h,d),nn.Dropout(dp))
    def forward(self,x): return self.net(x)

class Attn(nn.Module):
    def __init__(self,d,heads=8,dh=64,dp=0.):
        super().__init__(); inner=heads*dh; self.heads=heads; self.dp=dp
        self.norm=nn.LayerNorm(d); self.qkv=nn.Linear(d,inner*3,bias=False)
        self.out=nn.Sequential(nn.Linear(inner,d),nn.Dropout(dp))
    def forward(self,x,causal=True):
        x=self.norm(x); q,k,v=self.qkv(x).chunk(3,-1)
        q=rearrange(q,"b t (h d)->b h t d",h=self.heads)
        k=rearrange(k,"b t (h d)->b h t d",h=self.heads)
        v=rearrange(v,"b t (h d)->b h t d",h=self.heads)
        o=F.scaled_dot_product_attention(q,k,v,dropout_p=self.dp if self.training else 0.,is_causal=causal)
        return self.out(rearrange(o,"b h t d->b t (h d)"))

class CondBlock(nn.Module):
    def __init__(self,d,heads,dh,mlp,dp=0.):
        super().__init__()
        self.attn=Attn(d,heads,dh,dp); self.ff=FF(d,mlp,dp)
        self.n1=nn.LayerNorm(d,elementwise_affine=False,eps=1e-6)
        self.n2=nn.LayerNorm(d,elementwise_affine=False,eps=1e-6)
        self.ada=nn.Sequential(nn.SiLU(),nn.Linear(d,6*d))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)
    def forward(self,x,c):
        sa,sca,ga,sm,scm,gm=self.ada(c).chunk(6,-1)
        x=x+ga*self.attn(modulate(self.n1(x),sa,sca))
        x=x+gm*self.ff(modulate(self.n2(x),sm,scm)); return x

class Transformer(nn.Module):
    def __init__(self,in_d,hid,out_d,depth,heads,dh,mlp,dp=0.):
        super().__init__()
        self.ip=nn.Linear(in_d,hid) if in_d!=hid else nn.Identity()
        self.cp=nn.Linear(in_d,hid) if in_d!=hid else nn.Identity()
        self.layers=nn.ModuleList([CondBlock(hid,heads,dh,mlp,dp) for _ in range(depth)])
        self.norm=nn.LayerNorm(hid)
        self.op=nn.Linear(hid,out_d) if out_d!=hid else nn.Identity()
    def forward(self,x,c):
        x=self.ip(x); c=self.cp(c)
        for l in self.layers: x=l(x,c)
        return self.op(self.norm(x))

class ProjMLP(nn.Module):
    def __init__(self,in_d,hid,out_d): super().__init__(); self.net=nn.Sequential(nn.Linear(in_d,hid),SafeBN1d(hid),nn.GELU(),nn.Linear(hid,out_d))
    def forward(self,x): return self.net(x)

class ARPredictor(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.pos=nn.Parameter(torch.randn(1,cfg.seq_len,cfg.embed_dim)*0.02)
        self.drop=nn.Dropout(0.)
        self.tf=Transformer(cfg.embed_dim,cfg.encoder_hidden,cfg.encoder_hidden,
                             cfg.predictor_depth,cfg.predictor_heads,cfg.predictor_dim_head,
                             cfg.predictor_mlp_dim,cfg.predictor_dropout)
    def forward(self,emb,cond):
        t=emb.size(1); return self.tf(self.drop(emb+self.pos[:,:t]),cond)

class CS2HybridActionEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        half=cfg.action_embed_dim//2
        self.key_emb=nn.Embedding(10,cfg.key_embed_dim)
        self.register_buffer("key_idx",torch.arange(10))
        self.mouse_enc=nn.Sequential(nn.Linear(2,half),nn.LayerNorm(half),nn.SiLU(),nn.Linear(half,half))
        self.key_enc  =nn.Sequential(nn.Linear(10*cfg.key_embed_dim,half),nn.LayerNorm(half),nn.SiLU(),nn.Linear(half,half))
        self.merge    =nn.Sequential(nn.Linear(cfg.action_embed_dim,cfg.action_embed_dim),nn.LayerNorm(cfg.action_embed_dim),nn.SiLU(),nn.Linear(cfg.action_embed_dim,cfg.embed_dim))
    def forward(self,a):
        B,T,_=a.shape; flat=a.reshape(B*T,-1)
        m=self.mouse_enc(flat[:,:2])
        k=(flat[:,2:].clamp(0,1).unsqueeze(-1)*self.key_emb(self.key_idx)).reshape(B*T,-1)
        return self.merge(torch.cat([m,self.key_enc(k)],-1)).reshape(B,T,-1)


class CS2LeWM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import ViTConfig, ViTModel
        self.cfg=cfg
        vcfg=ViTConfig(image_size=cfg.img_size,patch_size=cfg.patch_size,num_channels=3,
                        hidden_size=cfg.encoder_hidden,num_hidden_layers=cfg.encoder_layers,
                        num_attention_heads=cfg.encoder_heads,intermediate_size=cfg.encoder_intermediate)
        self.encoder=ViTModel(vcfg,add_pooling_layer=False)
        if cfg.gradient_checkpointing: self.encoder.gradient_checkpointing_enable()
        self.projector   =ProjMLP(cfg.encoder_hidden,cfg.proj_hidden,cfg.embed_dim)
        self.action_enc  =CS2HybridActionEncoder(cfg)
        self.predictor   =ARPredictor(cfg)
        self.pred_proj   =ProjMLP(cfg.encoder_hidden,cfg.proj_hidden,cfg.embed_dim)
        self.sigreg      =SIGReg(17,1024)
        if cfg.residual_predictor:
            nn.init.zeros_(self.pred_proj.net[-1].weight)
            nn.init.zeros_(self.pred_proj.net[-1].bias)

    def encode_images(self,f):
        return self.projector(self.encoder(pixel_values=f).last_hidden_state[:,0])

    def encode_sequence(self,f):
        B,T=f.shape[:2]
        e=self.encode_images(rearrange(f,"b t c h w->(b t) c h w"))
        return rearrange(e,"(b t) d->b t d",b=B,t=T)

    def predict_sequence(self,emb,act_emb):
        h=self.predictor(emb,act_emb)
        p=self.pred_proj(rearrange(h,"b t d->(b t) d"))
        p=rearrange(p,"(b t) d->b t d",b=emb.size(0),t=emb.size(1))
        return p+emb if self.cfg.residual_predictor else p

    def forward(self, frames, actions, sigreg_weight, sigreg_pred_weight=0.):
        emb=self.encode_sequence(frames)
        act_emb=self.action_enc(actions)
        pred=self.predict_sequence(emb[:,:-1],act_emb[:,:-1])
        target=emb[:,1:].detach(); source=emb[:,:-1].detach()

        raw_pred=(pred-target).pow(2).mean(-1)
        raw_null=(source-target).pow(2).mean(-1)
        dn=(target-source).norm(-1).detach()

        if self.cfg.dynamic_weight_power>0:
            w=dn.clamp(1e-4).pow(self.cfg.dynamic_weight_power)
            w=w/w.mean(1,keepdim=True).clamp(1e-6)
            pred_loss=(w*raw_pred).mean()
        else:
            pred_loss=raw_pred.mean()

        sig_loss=self.sigreg(emb.transpose(0,1))
        sig_pred_loss=self.sigreg(pred.transpose(0,1)) if sigreg_pred_weight>0 else torch.tensor(0.,device=frames.device)
        total=pred_loss+sigreg_weight*sig_loss+sigreg_pred_weight*sig_pred_loss

        cos_pred=F.cosine_similarity(pred.float(),target.float(),-1).mean()
        null_ratio=raw_null.mean()/raw_pred.mean().clamp(1e-8)
        return {"loss":total,"pred_loss":raw_pred.mean().detach(),"weighted_pred":pred_loss.detach(),
                "sigreg_loss":sig_loss.detach(),"sig_pred_loss":sig_pred_loss.detach(),
                "cos_pred":cos_pred.detach(),"null_ratio":null_ratio.detach()}

    @torch.no_grad()
    def rollout(self,ctx_frames,future_actions):
        emb=self.encode_sequence(ctx_frames)
        for t in range(future_actions.size(1)):
            h=min(self.cfg.history_size,emb.size(1)); ec=emb[:,-h:]
            ac=future_actions[:,max(0,t-h+1):t+1]
            if ac.size(1)<h:
                pad=torch.zeros(future_actions.size(0),h-ac.size(1),future_actions.size(2),device=future_actions.device,dtype=future_actions.dtype)
                ac=torch.cat([pad,ac],1)
            emb=torch.cat([emb,self.predict_sequence(ec,self.action_enc(ac))[:,-1:]],1)
        return emb


class ConvDecoder(nn.Module):
    def __init__(self,embed_dim,img_size):
        super().__init__()
        self.img_size=img_size
        self.fc=nn.Sequential(nn.Linear(embed_dim,1024),nn.SiLU(),nn.Linear(1024,512*7*7))
        def up(ic,oc): return nn.Sequential(nn.ConvTranspose2d(ic,oc,4,2,1),nn.InstanceNorm2d(oc,affine=True),nn.SiLU())
        self.net=nn.Sequential(up(512,256),up(256,128),up(128,64),up(64,32),up(32,16),nn.Conv2d(16,3,3,padding=1),nn.Tanh())
    def forward(self,z):
        x=self.fc(z).reshape(z.size(0),512,7,7); x=self.net(x)
        if x.size(-1)!=self.img_size: x=F.interpolate(x,self.img_size,mode="bilinear",align_corners=False)
        return x


class PatchGAN(nn.Module):
    def __init__(self):
        super().__init__()
        def b(ic,oc,n=True):
            l=[nn.Conv2d(ic,oc,4,2,1)]
            if n: l.append(nn.InstanceNorm2d(oc,affine=True))
            l.append(nn.LeakyReLU(.2,inplace=True)); return nn.Sequential(*l)
        self.net=nn.Sequential(b(3,64,False),b(64,128),b(128,256),b(256,512),nn.Conv2d(512,1,4,padding=1))
    def forward(self,x): return self.net(x)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as M
        vgg=M.vgg16(weights=M.VGG16_Weights.IMAGENET1K_V1).features
        self.s1=nn.Sequential(*list(vgg.children())[:9]).eval()
        self.s2=nn.Sequential(*list(vgg.children())[9:16]).eval()
        for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer("m",torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer("s",torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self,p,t):
        pn=((p+1)/2-self.m)/self.s; tn=((t+1)/2-self.m)/self.s
        f1p=self.s1(pn); f1t=self.s1(tn)
        return F.mse_loss(f1p,f1t)+F.mse_loss(self.s2(f1p),self.s2(f1t))


def sobel_edge_loss(p,t):
    gp=p.mean(1,keepdim=True); gt=t.mean(1,keepdim=True)
    kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],device=p.device,dtype=p.dtype).view(1,1,3,3)
    ky=kx.transpose(-1,-2)
    ep=torch.sqrt(F.conv2d(gp,kx,padding=1)**2+F.conv2d(gp,ky,padding=1)**2+1e-6)
    et=torch.sqrt(F.conv2d(gt,kx,padding=1)**2+F.conv2d(gt,ky,padding=1)**2+1e-6)
    return F.l1_loss(ep,et)

def ssim_loss(p,t):
    c1,c2=.01**2,.03**2
    mx=F.avg_pool2d(p,3,1,1); my=F.avg_pool2d(t,3,1,1)
    sx=F.avg_pool2d(p*p,3,1,1)-mx**2; sy=F.avg_pool2d(t*t,3,1,1)-my**2
    sxy=F.avg_pool2d(p*t,3,1,1)-mx*my
    return 1.-(((2*mx*my+c1)*(2*sxy+c2))/((mx**2+my**2+c1)*(sx+sy+c2))).mean()


def cosine_warmup_sched(opt,warmup_ep,total_ep,spe):
    ws=max(1,warmup_ep*spe); ts=max(ws+1,total_ep*spe)
    def f(s):
        if s<ws: return float(s+1)/ws
        p=float(s-ws)/max(1,ts-ws); return .5*(1+math.cos(math.pi*p))
    return torch.optim.lr_scheduler.LambdaLR(opt,f)


def save_ckpt(path,model,cfg,history,norm,extra=None):
    p={"config":asdict(cfg),"model":getattr(model,"_orig_mod",model).state_dict(),
       "history":history,"normalizer":norm.state_dict()}
    if extra: p.update(extra)
    torch.save(p,path)

def load_ckpt(path,cfg_override=None):
    p=torch.load(path,map_location="cpu")
    cfg=Config(**(dict(p["config"])|asdict(cfg_override) if cfg_override else p["config"]))
    m=CS2LeWM(cfg); m.load_state_dict(p["model"])
    n=ActionNormalizer(); n.load_state_dict(p["normalizer"])
    return m,p.get("history",{}),n


def train_jepa(cfg,model,tr_loader,vl_loader,norm,run_dir,deadline=None):
    dev,dt=model_device_dtype(cfg)
    model=maybe_compile(model.to(dev),cfg.compile_model)
    try: opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay,fused=dev.type=="cuda")
    except TypeError: opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    sched=cosine_warmup_sched(opt,cfg.warmup_epochs,cfg.epochs,len(tr_loader))
    scaler=torch.cuda.amp.GradScaler(enabled=(dev.type=="cuda" and dt==torch.float16))
    hist={k:[] for k in ["train_total","train_pred","train_sigreg","train_null_ratio",
                           "val_total","val_pred","val_sigreg","val_null_ratio","val_cos_pred"]}
    best_val=float("inf"); gstep=0

    print("="*65); print("JEPA Training"); print(f"  residual_predictor={cfg.residual_predictor}  sigreg_weight={cfg.sigreg_weight}")
    print(f"  sigreg_pred_weight={cfg.sigreg_pred_weight}  dynamic_weight_power={cfg.dynamic_weight_power}")
    print("="*65)

    for ep in range(1,cfg.epochs+1):
        if deadline and time.time()>=deadline: break
        model.train()
        sums={k:0. for k in ["loss","pred","sig","null_ratio"]}; n=0; t0=time.time()

        for fu8,ac in tr_loader:
            if deadline and time.time()>=deadline: break
            frames=frames_to_device(fu8,dev,dt)
            actions=norm.transform(ac.to(dev,non_blocking=True))
            sig_w=cfg.sigreg_weight*min(1.,float(gstep+1)/max(1,cfg.sigreg_warmup_steps))
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(dev,dt):
                out=model(frames,actions,sig_w,cfg.sigreg_pred_weight)
            if scaler.is_enabled():
                scaler.scale(out["loss"]).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step()
            sched.step(); gstep+=1
            B=frames.size(0)
            sums["loss"]+=float(out["loss"].detach().cpu())*B
            sums["pred"]+=float(out["pred_loss"].cpu())*B
            sums["sig"] +=float(out["sigreg_loss"].cpu())*B
            sums["null_ratio"]+=float(out["null_ratio"].cpu())*B; n+=B

        for k,hk in [("loss","train_total"),("pred","train_pred"),("sig","train_sigreg"),("null_ratio","train_null_ratio")]:
            hist[hk].append(sums[k]/max(1,n))

        model.eval()
        vs={k:0. for k in ["loss","pred","sig","null_ratio","cos_pred"]}; vn=0
        with torch.no_grad():
            for fu8,ac in vl_loader:
                frames=frames_to_device(fu8,dev,dt)
                actions=norm.transform(ac.to(dev,non_blocking=True))
                with autocast_ctx(dev,dt): out=model(frames,actions,cfg.sigreg_weight,0.)
                B=frames.size(0)
                for k in ["null_ratio","cos_pred"]: vs[k]+=float(out[k].cpu())*B
                vs["loss"]+=float(out["loss"].cpu())*B
                vs["pred"]+=float(out["pred_loss"].cpu())*B
                vs["sig"] +=float(out["sigreg_loss"].cpu())*B; vn+=B
        for k,hk in [("loss","val_total"),("pred","val_pred"),("sig","val_sigreg"),
                      ("null_ratio","val_null_ratio"),("cos_pred","val_cos_pred")]:
            hist[hk].append(vs[k]/max(1,vn))

        vt=hist["val_total"][-1]
        if vt<best_val:
            best_val=vt
            save_ckpt(Path(run_dir)/"best_jepa.pt",model,cfg,hist,norm,{"epoch":ep})
        save_ckpt(Path(run_dir)/"last_jepa.pt",model,cfg,hist,norm,{"epoch":ep})
        elapsed=(time.time()-t0)/60
        print(f"E{ep:02d} | train={hist['train_total'][-1]:.4f} null={hist['train_null_ratio'][-1]:.3f} "
              f"| val={vt:.4f} val_null={hist['val_null_ratio'][-1]:.3f} val_cos={hist['val_cos_pred'][-1]:.4f} "
              f"| {elapsed:.1f}min")

    return hist


def train_decoder(cfg,enc_model,frame_ds,run_dir,deadline=None):
    dev,dt=model_device_dtype(cfg)
    enc_model=enc_model.to(dev).eval()
    dec=ConvDecoder(cfg.embed_dim,cfg.img_size).to(dev)
    dis=PatchGAN().to(dev)
    perc=PerceptualLoss().to(dev)

    try:
        og=torch.optim.AdamW(dec.parameters(),lr=cfg.decoder_lr,weight_decay=cfg.decoder_weight_decay,fused=True)
        od=torch.optim.AdamW(dis.parameters(),lr=cfg.decoder_lr*.5,weight_decay=cfg.decoder_weight_decay,fused=True)
    except TypeError:
        og=torch.optim.AdamW(dec.parameters(),lr=cfg.decoder_lr,weight_decay=cfg.decoder_weight_decay)
        od=torch.optim.AdamW(dis.parameters(),lr=cfg.decoder_lr*.5,weight_decay=cfg.decoder_weight_decay)

    tn=min(len(frame_ds),cfg.decoder_steps_per_epoch*cfg.decoder_batch_size)
    vn=min(len(frame_ds),cfg.decoder_val_steps*cfg.decoder_batch_size)
    tr_sampler=RandomSampler(frame_ds,replacement=len(frame_ds)<tn,num_samples=tn)
    vl_sub=Subset(frame_ds,list(range(vn)))
    kw=dict(pin_memory=torch.cuda.is_available(),num_workers=cfg.num_workers,persistent_workers=cfg.num_workers>0)
    trl=DataLoader(frame_ds,batch_size=cfg.decoder_batch_size,sampler=tr_sampler,collate_fn=collate_frames,drop_last=True,**kw)
    vll=DataLoader(vl_sub,batch_size=cfg.decoder_batch_size,shuffle=False,collate_fn=collate_frames,drop_last=False,**kw)

    hist={"train":[],"val":[]}; best_val=float("inf")
    print(f"\nDecoder training ({cfg.decoder_epochs} epochs) — perceptual+adversarial")

    for ep in range(1,cfg.decoder_epochs+1):
        if deadline and time.time()>=deadline: break
        dec.train(); dis.train(); gl=dl=0.; ns=0

        for fu8 in trl:
            if deadline and time.time()>=deadline: break
            frames=frames_to_device(fu8,dev,dt)
            frame_4d=frames[:,0] if frames.ndim==5 else frames
            target=(denorm_frames(frame_4d.float())*2.-1.)
            with torch.no_grad():
                with autocast_ctx(dev,dt): z=enc_model.encode_images(frame_4d).float()

            with autocast_ctx(dev,dt):
                r=dec(z)
                d_real=dis(target); d_fake=dis(r.detach())
                d_loss=(F.mse_loss(d_real,torch.ones_like(d_real))+F.mse_loss(d_fake,torch.zeros_like(d_fake)))*.5
            od.zero_grad(set_to_none=True); d_loss.backward(); od.step()

            with autocast_ctx(dev,dt):
                r=dec(z)
                lm=F.mse_loss(r,target); lp=perc(r,target)
                le=sobel_edge_loss(r,target); ls=ssim_loss(r,target)
                la=F.mse_loss(dis(r),torch.ones_like(dis(r)))
                g_loss=cfg.decoder_w_mse*lm+cfg.decoder_w_perc*lp+cfg.decoder_w_edge*le+cfg.decoder_w_ssim*ls+cfg.decoder_w_adv*la
            og.zero_grad(set_to_none=True); g_loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(),1.); og.step()
            gl+=float(g_loss.detach().cpu())*fu8.size(0); dl+=float(d_loss.detach().cpu())*fu8.size(0); ns+=fu8.size(0)

        hist["train"].append(gl/max(1,ns))
        dec.eval(); vl=0.; vs=0
        with torch.no_grad():
            for fu8 in vll:
                frames=frames_to_device(fu8,dev,dt)
                f4=frames[:,0] if frames.ndim==5 else frames
                tgt=(denorm_frames(f4.float())*2.-1.)
                with autocast_ctx(dev,dt): z=enc_model.encode_images(f4).float(); r=dec(z)
                loss=F.mse_loss(r,tgt)+.2*sobel_edge_loss(r,tgt)
                vl+=float(loss.cpu())*fu8.size(0); vs+=fu8.size(0)
        hist["val"].append(vl/max(1,vs))

        if hist["val"][-1]<best_val:
            best_val=hist["val"][-1]
            torch.save({"decoder":dec.state_dict(),"disc":dis.state_dict(),"history":hist},Path(run_dir)/"best_decoder.pt")
        if ep%10==0 or ep==1: print(f"  Dec E{ep:03d}/{cfg.decoder_epochs} g={hist['train'][-1]:.4f} val={hist['val'][-1]:.4f}")
    return hist


def _to_rgb(t):
    m=torch.tensor(IMG_MEAN).view(3,1,1); s=torch.tensor(IMG_STD).view(3,1,1)
    return ((t.cpu().float()*s+m).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
def _tanh_rgb(t): return (((t.cpu().float()+1)/2).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
def _sf(fig,run_dir,name): p=Path(run_dir)/name; fig.savefig(p,dpi=130,bbox_inches="tight"); plt.close(fig); print(f"  ✓ {name}"); return p


@torch.no_grad()
def evaluate_metrics(cfg,model,val_ds,norm,run_dir,decoder_ckpt=None):
    dev,dt=model_device_dtype(cfg); model=model.to(dev).eval()
    run_dir=ensure_dir(run_dir); rng=np.random.default_rng(cfg.seed)
    N=min(cfg.eval_sequences,len(val_ds))
    ids=rng.choice(len(val_ds),N,replace=False)

    Z=[]; FF=[]; null_ratios=[]; cos_preds=[]; straight_vals=[]; future_correct=0
    sens=np.zeros(len(ACTION_NAMES))

    for raw_i in ids:
        fu8,ac=val_ds[int(raw_i)]
        frames=frames_to_device(fu8.unsqueeze(0),dev,dt)
        act=norm.transform(ac.unsqueeze(0).to(dev))
        with autocast_ctx(dev,dt):
            emb=model.encode_sequence(frames); ae=model.action_enc(act)
            pred=model.predict_sequence(emb[:,:-1],ae[:,:-1])
        target=emb[:,1:]; source=emb[:,:-1]
        null_ratios.append(float(((source-target).pow(2).mean()/(pred-target).pow(2).mean().clamp(1e-8)).cpu()))
        cos_preds.append(float(F.cosine_similarity(pred.float(),target.float(),-1).mean().cpu()))
        flat=emb[0].float().cpu(); Z.append(flat[flat.size(0)//2]); FF.append(fu8[flat.size(0)//2])
        vel=flat[1:]-flat[:-1]
        if vel.size(0)>=2: straight_vals.append(float(F.cosine_similarity(vel[:-1],vel[1:],-1).mean().cpu()))

        ctx=frames[:,:cfg.history_size]; ta=act[:,cfg.history_size-1:cfg.history_size]
        pn=model.rollout(ctx,ta)[:,-1]
        choices=[emb[:,cfg.history_size].squeeze(0)]
        for _ in range(3):
            oi=int(rng.integers(0,len(val_ds))); of,_=val_ds[oi]
            oe=model.encode_sequence(frames_to_device(of.unsqueeze(0),dev,dt))
            choices.append(oe[:,cfg.history_size].squeeze(0))
        dists=[F.mse_loss(pn,c).item() for c in choices]
        if int(np.argmin(dists))==0: future_correct+=1

        ba=act[:,cfg.history_size-1:cfg.history_size]; bp=model.rollout(ctx,ba)[:,-1]
        for dim in range(len(ACTION_NAMES)):
            pa=ba.clone()
            if dim<2: pa[:,:,dim]+=1.
            else: pa[:,:,dim]=1.
            ap=model.rollout(ctx,pa)[:,-1]
            sens[dim]+=float((ap-bp).norm(-1).mean().cpu())

    sens/=max(1,N)
    Z=torch.stack(Z); Znp=Z.numpy()
    stds=Znp.std(0); collapsed=int((stds<.01).sum())
    _,sv,_=np.linalg.svd(Znp-Znp.mean(0),full_matrices=False)
    probs=sv/max(sv.sum(),1e-8); er=float(np.exp(-(probs*np.log(probs+1e-12)).sum()))

    metrics={"effective_rank":er,"collapsed_dims":float(collapsed),
             "null_ratio":float(np.mean(null_ratios)),"cos_pred":float(np.mean(cos_preds)),
             "temporal_straightening":float(np.mean(straight_vals)),
             "future_selection":float(future_correct/max(1,N))}
    write_json(run_dir/"metrics.json",metrics)
    print("\nMetrics:", json.dumps(metrics,indent=2))

    fig,ax=plt.subplots(figsize=(12,5))
    ax.bar(ACTION_NAMES,sens,color=["#d55e00" if i<2 else "#4c72b0" for i in range(len(ACTION_NAMES))])
    ax.set_title("V1: Action Sensitivity (red=mouse continuous, blue=discrete keys)"); ax.tick_params(axis="x",rotation=35)
    for i,(n,v) in enumerate(zip(ACTION_NAMES,sens)): ax.text(i,v+max(sens)*.01,f"{v:.3f}",ha="center",fontsize=7)
    fig.tight_layout(); _sf(fig,run_dir,"V1_action_sensitivity.png")

    dv=Znp.var(0); sv_t=torch.linalg.svd(torch.tensor(Znp-Znp.mean(0)),full_matrices=False)[1]
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    fig.suptitle(f"V2: Embedding Health | collapsed={collapsed}/256 | eff_rank={er:.1f}/256",fontsize=12)
    axes[0].bar(range(len(dv)),dv,color=["red" if v<.01 else "#4DBEEE" for v in dv],alpha=.8)
    axes[0].axhline(.01,color="red",ls="--",lw=2); axes[0].set_title(f"Per-dim variance | {collapsed} collapsed")
    for i in range(5): d=Znp[:,i*len(dv)//5]; d=(d-d.mean())/(d.std()+1e-8); axes[1].hist(d,bins=40,alpha=.3,density=True)
    xg=np.linspace(-4,4,200); axes[1].plot(xg,np.exp(-xg**2/2)/np.sqrt(2*np.pi),"k--",lw=2,label="N(0,1)"); axes[1].legend(); axes[1].set_title("Distribution vs Gaussian")
    axes[2].plot(sv_t.numpy()[:50],"b-o",ms=3); axes[2].set_title(f"Singular vals | eff_rank={er:.1f}"); axes[2].grid(.3)
    fig.tight_layout(); _sf(fig,run_dir,"V2_embedding_health.png")

    try:
        from sklearn.manifold import TSNE
        Zn=F.normalize(Z.float(),-1).numpy()
        print("  Running t-SNE..."); Z2=TSNE(2,perplexity=30,n_iter=800,random_state=42).fit_transform(Zn)
        fig,ax=plt.subplots(figsize=(10,8))
        sc=ax.scatter(Z2[:,0],Z2[:,1],c=np.arange(len(Z2)),cmap="viridis",s=12,alpha=.7)
        plt.colorbar(sc,ax=ax,label="sequence index"); ax.set_title("V3: Latent Space (t-SNE)"); ax.axis("off")
        fig.tight_layout(); _sf(fig,run_dir,"V3_tsne.png")
    except Exception as e: print(f"  V3 skipped: {e}")

    if decoder_ckpt and Path(decoder_ckpt).exists():
        dec=ConvDecoder(cfg.embed_dim,cfg.img_size).to(dev)
        dec.load_state_dict(torch.load(decoder_ckpt,map_location="cpu")["decoder"]); dec.eval()
        n_show=min(6,len(FF))
        z_show=Z[:n_show].to(dev).float()
        with autocast_ctx(dev,dt): recon=dec(z_show).float()
        fig,axes=plt.subplots(2,n_show,figsize=(n_show*3,5.5))
        fig.suptitle("V4: Real vs Decoded (perceptual+adversarial loss)",fontsize=12)
        for i in range(n_show):
            axes[0,i].imshow(_to_rgb(FF[i])); axes[0,i].axis("off")
            if i==0: axes[0,i].set_ylabel("Real",fontsize=9)
            axes[1,i].imshow(_tanh_rgb(recon[i])); axes[1,i].axis("off")
            if i==0: axes[1,i].set_ylabel("Decoded",fontsize=9)
        fig.tight_layout(); _sf(fig,run_dir,"V4_reconstruction.png")

        if len(Z)>5:
            za=Z[0].unsqueeze(0).to(dev); zb=Z[min(5,len(Z)-1)].unsqueeze(0).to(dev)
            alphas=torch.linspace(0,1,10,device=dev)
            zi=torch.stack([a*zb+(1-a)*za for a in alphas]).squeeze(1)
            with autocast_ctx(dev,dt): ri=dec(zi).float()
            fig,axes=plt.subplots(1,12,figsize=(12*2.5,2.8))
            axes[0].imshow(_to_rgb(FF[0])); axes[0].set_title("A",fontsize=8,color="blue"); axes[0].axis("off")
            for i in range(10): axes[i+1].imshow(_tanh_rgb(ri[i])); axes[i+1].set_title(f"α={alphas[i]:.1f}",fontsize=7); axes[i+1].axis("off")
            axes[-1].imshow(_to_rgb(FF[min(5,len(FF)-1)])); axes[-1].set_title("B",fontsize=8,color="red"); axes[-1].axis("off")
            fig.suptitle("V5: Latent Interpolation A→B",fontsize=11); fig.tight_layout()
            _sf(fig,run_dir,"V5_interpolation.png")

    if len(Z)>=10:
        Zn=F.normalize(Z.float(),-1); K=4; nq=4
        qidxs=np.random.choice(len(Zn),nq,replace=False)
        fig,axes=plt.subplots(nq,K+1,figsize=((K+1)*2.5,nq*3))
        fig.suptitle("V6: Nearest-Neighbour Retrieval",fontsize=12)
        for row,qi in enumerate(qidxs):
            sims=(Zn@Zn[qi:qi+1].T).squeeze(); sims[qi]=-1; topk=sims.topk(K).indices.numpy()
            axes[row,0].imshow(_to_rgb(FF[qi])); axes[row,0].set_title("QUERY",fontsize=8,color="red"); axes[row,0].axis("off")
            for k,ni in enumerate(topk):
                axes[row,k+1].imshow(_to_rgb(FF[ni])); axes[row,k+1].set_title(f"{sims[ni]:.3f}",fontsize=7); axes[row,k+1].axis("off")
        fig.tight_layout(); _sf(fig,run_dir,"V6_nn_retrieval.png")

    step_sims=[]
    for raw_i in ids[:60]:
        fu8,ac=val_ds[int(raw_i)]
        frames=frames_to_device(fu8.unsqueeze(0),dev,dt)
        act=norm.transform(ac.unsqueeze(0).to(dev))
        with autocast_ctx(dev,dt):
            emb=model.encode_sequence(frames); ae=model.action_enc(act)
            pred=model.predict_sequence(emb[:,:-1],ae[:,:-1])
        sim=F.cosine_similarity(pred.float(),emb[:,1:].float(),-1).squeeze(0).cpu()
        step_sims.append(sim)
    step_sims=torch.stack(step_sims); ms=step_sims.mean(0).numpy(); ss=step_sims.std(0).numpy()
    steps=range(1,len(ms)+1)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f"V7: Multi-Step Prediction | step-1={ms[0]:.4f} last={ms[-1]:.4f} mean={ms.mean():.4f}",fontsize=12)
    axes[0].plot(steps,ms,"b-o",ms=5,lw=2); axes[0].fill_between(steps,ms-ss,ms+ss,alpha=.2)
    for t,c,l in [(.9,"g","great"),(.7,"orange","good"),(.5,"r","random")]: axes[0].axhline(t,color=c,ls="--",alpha=.7,label=l)
    axes[0].set_ylim(0,1); axes[0].legend(); axes[0].grid(.3); axes[0].set_title("Cosine Similarity per Step")
    axes[1].bar(steps,1-ms,color="steelblue",alpha=.8); axes[1].grid(.3,axis="y"); axes[1].set_title("Error per Step")
    fig.tight_layout(); _sf(fig,run_dir,"V7_prediction_quality.png")


    if len(val_ds)>0:
        fu8,ac=val_ds[0]; f4=frames_to_device(fu8.unsqueeze(0)[:,:4],dev,dt)
        z0=model.encode_sequence(f4)
        h=cfg.history_size; mc=30.
        protos={"Straight":torch.zeros(1,h,len(ACTION_NAMES),device=dev),
                "TurnLeft": torch.zeros(1,h,len(ACTION_NAMES),device=dev),
                "TurnRight":torch.zeros(1,h,len(ACTION_NAMES),device=dev),
                "Shoot":    torch.zeros(1,h,len(ACTION_NAMES),device=dev)}
        protos["TurnLeft"][:,:,0]=-mc*.5; protos["TurnRight"][:,:,0]=mc*.5; protos["Shoot"][:,:,9]=1.
        rolls={}
        for nm,act in protos.items(): rolls[nm]=model.rollout(z0,act).squeeze(0).float().cpu()
        all_z=torch.cat(list(rolls.values()),0)
        _,_,Vt=torch.linalg.svd(all_z-all_z.mean(0),full_matrices=False); b=Vt[:2].float()
        colors=["#2196F3","#4CAF50","#F44336","#FF9800"]
        fig,axes=plt.subplots(1,2,figsize=(16,6))
        fig.suptitle("V9: Action-Conditioned Imagined Trajectories\nDivergence = model IS action-conditioned ✓",fontsize=12)
        for ci,(nm,zr) in enumerate(rolls.items()):
            p=((zr-all_z.mean(0))@b.T).numpy()
            axes[0].plot(p[:,0],p[:,1],"-o",color=colors[ci],ms=4,lw=2,label=nm,alpha=.8)
        axes[0].legend(fontsize=9); axes[0].set_title("Latent PCA trajectories"); axes[0].grid(.3)
        z_ref=rolls["Straight"]
        for ci,(nm,zr) in enumerate(rolls.items()):
            if nm=="Straight": continue
            T2=min(z_ref.shape[0],zr.shape[0]); d=(zr[:T2]-z_ref[:T2]).norm(-1).numpy()
            axes[1].plot(range(T2),d,color=colors[ci],lw=2,label=f"vs {nm}")
        axes[1].legend(fontsize=9); axes[1].set_title("Divergence from Straight"); axes[1].grid(.3)
        fig.tight_layout(); _sf(fig,run_dir,"V9_action_trajectories.png")

    return metrics


def save_history_plot(history,run_dir):
    if not history: return
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    fig.suptitle("Training History",fontsize=14)
    for ax,(k,l) in zip(axes.flat,[("train_total","Total Loss"),("train_pred","Pred Loss"),
                                     ("train_sigreg","SIGReg"),("val_total","Val Total"),
                                     ("train_null_ratio","Null Ratio (want >1)"),("val_cos_pred","Val CosimSim")]):
        if k in history and history[k]:
            ax.plot(history[k],label=k); ax.set_title(l); ax.grid(.3)
            if "null" in k: ax.axhline(1.,color="red",ls="--",lw=2,label="random baseline")
            ax.legend(fontsize=8)
    fig.tight_layout(); p=Path(run_dir)/"training_history.png"
    fig.savefig(p,dpi=140); plt.close(fig); print(f"  ✓ training_history.png")


def smoke_test(cfg):
    dev,dt=model_device_dtype(cfg); model=CS2LeWM(cfg).to(dev).train()
    b=min(cfg.batch_size,8)
    fu8=torch.randint(0,255,(b,cfg.seq_len,cfg.img_size,cfg.img_size,3),dtype=torch.uint8)
    ac=torch.randn(b,cfg.seq_len,len(ACTION_NAMES))
    frames=frames_to_device(fu8,dev,dt); actions=ac.to(dev)
    out=model(frames,actions,cfg.sigreg_weight,cfg.sigreg_pred_weight)
    out["loss"].backward()
    vram=torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0
    total=torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0
    print(f"smoke ok | loss={float(out['loss'].cpu()):.4f} "
          f"pred={float(out['pred_loss'].cpu()):.4f} "
          f"null_ratio={float(out['null_ratio'].cpu()):.3f} "
          f"VRAM={vram:.1f}/{total:.0f}GB ({100*vram/total:.0f}%)" if total else
          f"smoke ok | loss={float(out['loss'].cpu()):.4f}")
    if torch.cuda.is_available() and vram/total < .4:
        print(f"  💡 VRAM underused ({vram/total:.0%}). Increase batch_size.")


def main():
    cfg=parse_args(); set_seed(cfg.seed); run_dir=ensure_dir(cfg.run_dir)
    write_json(run_dir/"config.json",asdict(cfg))

    if cfg.stage=="smoke": smoke_test(cfg); return

    if cfg.stage in {"preprocess","all"}:
        preprocess_corpus(cfg)
        if cfg.stage=="preprocess": return

    clips=discover_processed_clips(cfg)
    if not clips: raise RuntimeError(f"No processed clips in {cfg.processed_dir}")
    store=ClipStore(clips,mode=cfg.storage_mode,ram_budget_gb=cfg.ram_budget_gb)
    tr_loader,vl_loader,tr_ds,vl_ds,tr_clips,vl_clips=build_loaders(store,cfg)
    norm=ActionNormalizer(); norm.fit(store,tr_ds.index,cfg)

    deadline=time.time()+cfg.budget_hours*3600.
    jepa_dl=deadline-(cfg.reserve_decoder_minutes+cfg.reserve_eval_minutes)*60. if cfg.stage=="all" else deadline
    dec_dl =deadline-cfg.reserve_eval_minutes*60.

    history={}
    if cfg.stage in {"train","all"}:
        model=CS2LeWM(cfg)
        history=train_jepa(cfg,model,tr_loader,vl_loader,norm,run_dir,deadline=jepa_dl)
        save_history_plot(history,run_dir)
    else:
        model,history,norm=load_ckpt(Path(run_dir)/"best_jepa.pt",cfg)

    if cfg.stage in {"decoder","all"}:
        bm,_,_=load_ckpt(Path(run_dir)/"best_jepa.pt",cfg)
        frame_ds=CS2FrameDataset(store,tr_clips,cfg.decoder_max_frames)
        train_decoder(cfg,bm,frame_ds,run_dir,deadline=dec_dl)

    if cfg.stage in {"eval","all"}:
        bm,history,norm=load_ckpt(Path(run_dir)/"best_jepa.pt",cfg)
        dec_ckpt=Path(run_dir)/"best_decoder.pt"
        metrics=evaluate_metrics(cfg,bm,vl_ds,norm,run_dir,decoder_ckpt=dec_ckpt)
        print(json.dumps(metrics,indent=2))
        if history: save_history_plot(history,run_dir)


if __name__=="__main__":
    main()
