"""Freeze the complete pre-loop block state once, then permit normal denoising recurrence."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_tree(value, tensor_fn):
    import torch
    if isinstance(value,torch.Tensor): return tensor_fn(value)
    if isinstance(value,dict):
        if not all(isinstance(k,str) for k in value): raise TypeError("Only string-keyed state mappings supported")
        return {k:map_tree(v,tensor_fn) for k,v in value.items()}
    if isinstance(value,list): return [map_tree(v,tensor_fn) for v in value]
    if isinstance(value,tuple): return tuple(map_tree(v,tensor_fn) for v in value)
    if value is None or type(value) in (str,bool,int,float): return value
    raise TypeError("Unsupported state object: "+type(value).__name__)


def fingerprints(value):
    import torch
    if is_dataclass(value): value={f.name:getattr(value,f.name) for f in fields(value)}
    def one(t):
        t=t.detach().cpu().contiguous()
        raw=t.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(raw).hexdigest())
    return map_tree(value,one)


def restore_tree(saved, template):
    """Validate schema before moving frozen values to the current run's devices."""
    import torch
    if isinstance(saved,torch.Tensor):
        if not isinstance(template,torch.Tensor) or saved.shape!=template.shape or saved.dtype!=template.dtype:
            raise ValueError("Frozen tensor shape/dtype changed")
        return saved.to(device=template.device).clone()
    if isinstance(saved,dict):
        if not isinstance(template,dict) or saved.keys()!=template.keys(): raise ValueError("State keys changed")
        return {k:restore_tree(v,template[k]) for k,v in saved.items()}
    if isinstance(saved,(tuple,list)):
        if type(saved)!=type(template) or len(saved)!=len(template): raise ValueError("State sequence changed")
        return type(saved)(restore_tree(v,t) for v,t in zip(saved,template))
    if type(saved)!=type(template): raise ValueError("State scalar type changed")
    return saved


def generation_contract(args, model_paths=None):
    """Pin semantic arguments and input/config bytes; node and output paths may differ."""
    values=dict(vars(args))
    model=Path(values.pop("model_path"))
    for k in ("output","gpus","legacy_text_gpus","legacy_transformer_gpus","legacy_memory_fraction"):
        values.pop(k,None)
    for k in ("first_frame","last_frame","reference_image","reference_video","audio"):
        if values.get(k) is not None:
            p=Path(values[k]); values[k]=dict(bytes=p.stat().st_size,sha256=digest(p))
    configs={}
    modular=(model/"diffusers") if (model/"diffusers").is_dir() else model
    directories={"modular":modular}
    if model_paths is not None:
        directories={"modular":Path(model_paths.modular_dir),"partition":Path(model_paths.partition_dir)}
    for label,directory in directories.items():
        for name in ("modular_model_index.json","model_index.json","transformer/config.json","transformer_ref/config.json",
                     "text_encoder/config.json","vae/config.json","audio_vae/config.json",
                     "scheduler/scheduler_config.json","audio_scheduler/scheduler_config.json"):
            if (directory/name).is_file(): configs[label+"/"+name]=digest(directory/name)
    if model_paths is not None and not configs: raise ValueError("Resolved model configs were not fingerprinted")
    return dict(arguments=values,model_configs=configs,model_weights_hashed=False)


SCHEDULER_FIELDS=("timesteps","sigmas","_step_index","_begin_index","num_inference_steps")
REQUIRED=("latents","audio_latents","prompt_embeds","row_timestep_plan","timesteps","audio_timesteps",
          "denoiser_input_fields","num_condition_video_rows","num_condition_audio_rows")


class FrozenLoop:
    def __init__(self, mode, directory, report_dir, contract=None, budget_mib=512):
        if mode not in ("capture","replay"): raise ValueError("Unknown freeze mode")
        self.mode,self.directory,self.report_dir=mode,Path(directory),Path(report_dir)
        self.contract=contract
        self.budget=budget_mib*1024*1024
        if self.budget<=0: raise ValueError("Positive snapshot budget required")
        self.calls=0; self.steps=0; self.first=None; self.binding=None
        self.metadata=None
        self.log=None

    def _schedulers(self, components):
        result={}
        for name in ("scheduler","audio_scheduler"):
            owner=getattr(components,name)
            if any(not hasattr(owner,k) for k in SCHEDULER_FIELDS):
                raise ValueError("Unsupported scheduler state")
            result[name]=dict(kind=owner.__class__.__module__+"."+owner.__class__.__name__,
                              config=dict(owner.config),
                              state={k:getattr(owner,k) for k in SCHEDULER_FIELDS})
        return result

    def prepare(self, block_state, components):
        import torch
        if self.calls: raise RuntimeError("Exactly one denoising loop per process is supported")
        self.calls+=1
        state=dict(vars(block_state))
        if any(k not in state for k in REQUIRED): raise ValueError("Missing pre-loop fields; upstream API changed")
        current=dict(block=state,schedulers=self._schedulers(components))
        # Validate every object and budget in memory before serializing any large tensor.
        size=[0]
        def copy(t):
            size[0]+=t.numel()*t.element_size()
            if size[0]>self.budget: raise ValueError("Frozen state exceeds byte budget")
            return t.detach().cpu().clone() if self.mode=="capture" else t
        saved=map_tree(current,copy)
        if self.contract is None: raise RuntimeError("Generation contract was not bound")
        if self.mode=="capture":
            self.directory.mkdir(parents=True,exist_ok=False)
            path=self.directory/"upstream.pt"
            torch.save(saved,path)
            if path.stat().st_size>self.budget:
                path.unlink(); raise ValueError("Serialized state exceeds byte budget")
            self.metadata=dict(schema=1,complete=False,contract=self.contract,binding=self.binding,
                               snapshot_sha256=digest(path),snapshot_bytes=path.stat().st_size,
                               recurrence="restore once before loop; later latents evolve normally",
                               model_acceptance=False)
            self._save()
        else:
            self.metadata=json.loads((self.directory/"freeze.json").read_text())
            path=self.directory/"upstream.pt"
            if not self.metadata.get("complete") or self.metadata.get("schema")!=1:
                raise ValueError("Incomplete frozen generation")
            if self.metadata["contract"]!=self.contract or self.metadata["binding"]!=self.binding:
                raise ValueError("Upstream source, model config or generation contract changed")
            if path.stat().st_size>self.budget or digest(path)!=self.metadata["snapshot_sha256"]:
                raise ValueError("Frozen snapshot size/SHA mismatch")
            frozen=torch.load(path,map_location="cpu",weights_only=True)
            replacement=restore_tree(frozen,current)
            # Scheduler identity/config are validation gates, never replaced by a candidate's values.
            for name in ("scheduler","audio_scheduler"):
                if replacement["schedulers"][name]["kind"]!=current["schedulers"][name]["kind"] or (
                    fingerprints(replacement["schedulers"][name]["config"])!=fingerprints(current["schedulers"][name]["config"])):
                    raise ValueError("Scheduler identity/config changed")
            for key,value in replacement["block"].items(): setattr(block_state,key,value)
            for name in ("scheduler","audio_scheduler"):
                for k,v in replacement["schedulers"][name]["state"].items(): setattr(getattr(components,name),k,v)
        self.report_dir.mkdir(parents=True,exist_ok=True)
        self.log=(self.report_dir/"dit-inputs.jsonl").open("x",encoding="utf-8")
        return block_state

    def observe(self, inputs):
        if not self.calls or self.log is None: raise RuntimeError("DiT entered before frozen-loop hook")
        row=fingerprints(inputs)
        if self.steps==0:
            self.first=row
            if self.mode=="replay" and row!=self.metadata["first_dit_input"]:
                raise ValueError("First complete DiT input differs; refusing causal comparison")
            if self.mode=="capture":
                self.metadata["first_dit_input"]=row; self._save()
        self.log.write(json.dumps(dict(step=self.steps,inputs=row),sort_keys=True)+"\n"); self.log.flush()
        self.steps+=1

    def _save(self):
        path=self.directory/"freeze.json"
        temp=path.with_suffix(".json.partial")
        temp.write_text(json.dumps(self.metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        temp.replace(path)

    def close(self, complete):
        if self.log: self.log.close(); self.log=None
        valid=bool(complete and self.calls==1 and self.steps>0)
        if valid and self.mode=="replay" and self.steps!=self.metadata["dit_calls"]:
            valid=False
        if self.mode=="capture" and self.metadata:
            self.metadata.update(complete=valid,dit_calls=self.steps); self._save()
        (self.report_dir/"freeze-receipt.json").write_text(json.dumps(dict(
            complete=valid,mode=self.mode,source=str(self.directory),dit_calls=self.steps,
            first_input_equal=(valid if self.mode=="replay" else None),
            snapshot_sha256=self.metadata["snapshot_sha256"] if self.metadata else None,
            model_acceptance=False),indent=2)+"\n")
        if complete and not valid: raise RuntimeError("Frozen-loop execution incomplete or call count changed")

    def install(self, loop_cls, source_files):
        self.binding={name:digest(path) for name,path in sorted(source_files.items())}
        own="get_block_state" in loop_cls.__dict__
        original=loop_cls.get_block_state
        # components is only available in __call__, so bind it while the original loop gets its state.
        original_call=loop_cls.__call__
        active=[None]
        def get(owner,state):
            result=original(owner,state)
            if active[0] is None: raise RuntimeError("Loop state requested outside guarded call")
            return self.prepare(result,active[0])
        def call(owner,components,state):
            if active[0] is not None: raise RuntimeError("Nested denoising loops unsupported")
            active[0]=components
            try: return original_call(owner,components,state)
            finally: active[0]=None
        loop_cls.get_block_state=get; loop_cls.__call__=call
        def restore():
            loop_cls.__call__=original_call
            if own: loop_cls.get_block_state=original
            else: delattr(loop_cls,"get_block_state")
        return restore
