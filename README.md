This is a fork of original ERIQ repo to help evaluate reasoning abilities.  
Infer script added, feel free to use it!  
Training method of high level reasoning VLM is also included.

# HighLevel VLM Training

## Goal

Train a high-level VLM with generalization capabilities, building task decomposition and state monitoring abilities for long-horizon manipulation tasks.

## Data

HighLevel VLM is fundamentally a next-token-prediction paradigm for embodied scene understanding, so the data falls into three major categories:

> General Visual VQA Data
>
> Embodied Reasoning VQA
>
> Subtask Decomposition & State Monitoring Related Data

### Inventory of Available Data

#### General Visual VQA

General Visual VQA data refers to the generic data required for VLM fine-tuning, such as caption, grounding, and OCR. This category of data helps maintain the pre-trained checkpoint's visual understanding capability and prevents catastrophic forgetting during subsequent VLM fine-tuning.

This data is available in sufficient volume; at training time we only need to sample 10%–20% of the total training set. Currently we are using Cambrian737k.

#### Embodied Reasoning VQA

Embodied Reasoning VQA is designed to improve embodied reasoning abilities, covering dimensions such as affordance, pointing, physical commonsense, success detection, failure recovery, and more. This data contributes to both task decomposition and state monitoring capabilities.

| Dataset Name | # VQA Samples | Visual Input | Dimension Classification | Format Conversion | Training | Other Issues |
|---|---|---|---|---|---|---|
| EO-Data1.5M | 1.5M | Images | Yes | Yes | Yes | 0.3M samples contain interleaved-action data and are unusable; image resolution is low |
| RoboVQA | 508K | Video | Yes | Yes | Yes | The same video may have questions of different categories |
| ShareRobot | 616K | Image sequences | Yes | Yes | Yes | All visual data are ImageLists (avg. 10 images per question) |
| EgoIT-99K | 99K | Video or images | No | Yes | No | — |
| Magma | 8.3M | Video or images | No | No | No | Focuses on Mark; no language annotations |

> "Format Conversion" means whether the original heterogeneous formats have been converted to the standard JSONL data format locally.

#### Subtask Decomposition & State Monitoring Business Data

Business data refers to egocentric, UMI, and robot data that contains subtask annotations. This data requires secondary processing on top of the original source data to produce subtask decomposition and state monitoring training samples.

| Data Source | Scale | Visual Input | Sampling Strategy | Data Type | Processed | In Training | Other Issues |
|---|---|---|---|---|---|---|---|
| AgiBotWorld-Beta | 1M episodes | Images | Sample 10 episodes per task | Robot | Yes | Yes | Only 217 unique tasks; deduplication needed |
| genrobot | 1.1M subtasks / 2190 hrs | Fisheye images | Scene-balanced sampling | UMI | Yes | Yes | both open-soure and close-source |
| EgoExo4D | — | Video | — | Egocentric | Yes | No | — |
| Ego4D | — | Video | — | Egocentric | Yes | No | — |
| Xperience | — | Video | — | Egocentric | No | No | — |

### Annotation Method

As shown below, for each subtask boundary point ±200 ms we define a **Transition Phase**; all other regions are **Execution Phase**.

N frames (currently 5) are randomly sampled from each phase and annotated with VQA.

```
                                                    ░░ Transition Phase   ██ Execution Phase
Model           Last action is finished, the        continue the       Roll is on the table. The
Predictions     next action is to apply...          current task...    next action is to handover...
                                     ⬇                    ⬇                  ⬇
│░░░░█████████████████████████████░░░░░░░███████████████████████████████████░░░░░░░██████████████
└───────────────────────────────────┘─────────────────────────────────────────┘─────────────────┘
    remove bar code from label roll |apply bar code label, place roll on table|    handover
```

### VQA Annotation Format

#### Task Decomposition

```python
vqa_item = {
    "id": file_id,
    "image": image_paths,
    "source_mcap": source_mcap,
    "conversations": [
        {
            "from": "human",
            "value": (
                "<image>\n<image>\n<image>\n"
                "The observations are captured from ego view, left_wrist view and right_wrist view. "
                "In order to finish the task:{task} What should the robot do?"
            )
        },
        {
            "from": "gpt",
            "value": {subtask}
        }
    ]
}
```

#### State Monitoring

```python
vqa_item = {
    "id": file_id,
    "image": image_paths,
    "source_mcap": source_mcap,
    "conversations": [
        {
            "from": "human",
            "value": (
                "<image>\n<image>\n"
                "The two images represent the start of {subtask} observation and current observation "
                "from ego view respectively. What is the progress of {subtask} now?"
            )
        },
        {
            "from": "gpt",
            "value": "done or ongoing"
        }
    ]
}
```

## Model Fine-tuning

### QWEN3

Currently using Qwen3-4B for VQA LoRA fine-tuning, with offline evaluation on the ERIQ benchmark.

- https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
- https://github.com/GenieReasoner/ERIQ

### QWEN3.5

The official Qwen GitHub fine-tuning code only supports up to QWEN3. Fine-tuning QWEN3.5 requires switching to ms-swift.

- https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/Qwen3_5-Best-Practice.md

### Good dataset after finetuning 
Datasets below show improvements on ERIQ score when training my model with.  
These datasets have multiple annotations towards one scene, so try sampling to get better results.  
1、AgibotWorld-Beta:https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta  
2、EO-Date1.5M:https://huggingface.co/datasets/IPEC-COMMUNITY/EO-Data1.5M  
3、ROBOVQA:https://huggingface.co/datasets/Tianli/robovqa  

## Challenges

Subtask decomposition and progress understanding rely on context. The key question is: how to efficiently and cleanly inject context / memory into the model without hacks.
