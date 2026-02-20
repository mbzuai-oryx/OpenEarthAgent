# 🔧 Debug Guidelines

## Check Tool Server Logs

After starting the tool server, verify that it initialized correctly. Separate logs for each tool are available in the logs directory.

### 📍 Log Location

`OpenEarthAgent/tool_server/tool_workers/logs/server_log`

### ✅ What to Check

- Server startup errors  
- Missing dependencies  
- Configuration loading issues  
- Tool registration failures  
- Runtime exceptions  

If a tool is not responding, this log file should be your first checkpoint.

---
## Fix for `SegmentObjectPixels` Config Loading Issue
❌ Problem: Using `hydra.compose()` fails when a direct file path is provided.

✅ Recommended Fix: Update the logic to detect whether the input is a file path or a Hydra config name.

Replace

```python
cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
```

in `sam2/build_sam.py` with following: 

```python
if os.path.isfile(config_file):
    cfg = OmegaConf.load(config_file)
else:
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
```
---
# 3️⃣ Fix: ObjectDetection Config Issues

This section covers common configuration-related errors when running the `ObjectDetection` tool with LAE-DINO.

## Fix BERT Path Error
❌ Problem: You may encounter errors due to an invalid relative path for the BERT language model:

```python
lang_model_name = '../weights/bert-base-uncased'
```

✅ Solution: Modify the config file:
`OpenEarthAgent/models/LAE-DINO/mmdetection_lae/configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py`

Change: `lang_model_name = '../weights/bert-base-uncased'` to `lang_model_name = 'bert-base-uncased'`

Using: 'bert-base-uncased' allows HuggingFace Transformers to:

- Automatically download the model (if not cached)
- Load it from the local cache
- Resolve the correct path internally


## Fix: AttributeError: 'ConfigDict' object has no attribute 'pipeline'
❌ Problem: you may encounter the following error:

`AttributeError: 'ConfigDict' object has no attribute 'pipeline'`


This happens due to a mismatch between LAE-DINO config structure and MEngine / MMDetection version

✅ Recommended Fix: Modify the `_init_pipeline()` function in:

`OpenEarthAgent/models/LAE-DINO/mmdetection_lae/mmdet/apis/det_inferencer.py`

Replace:

```python
pipeline_cfg = cfg.test_dataloader.dataset.pipeline
```

With:
```python
# ---- FIX START ----
if hasattr(cfg.test_dataloader.dataset, 'pipeline'):
    pipeline_cfg = cfg.test_dataloader.dataset.pipeline
elif hasattr(cfg, 'test_pipeline'):
    pipeline_cfg = cfg.test_pipeline
else:
    raise AttributeError(
        'No test pipeline found in config. '
        'Expected either cfg.test_dataloader.dataset.pipeline '
        'or cfg.test_pipeline.'
    )
# ---- FIX END ----
```