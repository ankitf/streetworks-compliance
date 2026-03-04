# **Street Works Compliance using Cosmos Reason-2**
**Decision Reel: Spatiotemporal Barrier Continuity Assessment**

## Overview

This project explores whether a reasoning-oriented vision-language model can support structured compliance-style inspection tasks.

Rather than implementing full regulatory compliance, the notebook evaluates a focused hypothesis:

> **Cosmos Reason-2 enables structured spatiotemporal reasoning for inspection-oriented visual analysis tasks beyond simple object detection.**

The selected scenario is **barrier continuity assessment** in a street-works environment. The system detects structural discontinuities in temporary construction barriers using prompt-engineered inspection logic and produces audit-ready artefacts.

> Note:
>
> The accompanying notebook contains a more detailed discussion of the motivation, reasoning hypothesis, prompt design considerations, implementation trade-offs, and observed behaviors. The README provides a high-level summary, while the notebook documents the full exploratory workflow.

## **Scope**

Street-works compliance spans dozens of regulatory conditions and contextual rules. Implementing complete end-to-end compliance e.g., [Red Book (UK)](https://assets.publishing.service.gov.uk/media/5a7d8038e5274a676d532707/safety-at-streetworks.pdf) or [MUTCD (US)](https://mutcd.fhwa.dot.gov)  is intentionally out of scope.

This implementation focuses on a single representative task:

**Barrier Continuity and Effectiveness**

The goal is to detect:
* Visible gaps between adjacent barrier panels
* Missing or detached connections
* Breaks in the continuous physical boundary

The task requires:
* Spatiotemporal reasoning
* Structural continuity interpretation
* Functional assessment beyond object presence

## **Approach**

### **Prompt-Driven Domain Encoding**

No fine-tuning or custom training is used.

Compliance logic is encoded through structured prompts that:
* Define inspection intent
* Specify discontinuity criteria
* Constrain output to structured JSON

The model:
* Receives the full video
* Internally extracts frames
* Performs temporal reasoning
* Returns structured timestamps of detected discontinuities

### **Post-Processing Pipeline**

* Model output is transformed into operational artefacts through:
* Timestamp merging into continuous intervals
* Evidence frame extraction
* Decision reel generation (video summary of flagged intervals)
* Static collage generation (visual summary)
* Structured summary report

**This converts raw reasoning output into inspection-ready documentation.**

## **Observed Behaviors**

During development, several important behaviors were observed:

* Detection quality improves when prompts explicitly encode temporal continuity.
* Overly verbose reasoning instructions may reduce detection coverage (reasoning overload).
* Output granularity may vary slightly across runs due to decoding dynamics.
* Interval merging is essential for stable, audit-style reporting.
* Modular rule-specific inference is likely more scalable than a single monolithic compliance prompt.

These findings suggest that prompt calibration and aggregation logic are critical components of reasoning-based compliance systems.

## **Repository Structure**

* notebook/ – Main Colab notebook
* sample_data/ – Example inspection video
* sample_output/ – Example outputs (two representative runs)

```
street_works_compliance/
│
├── README.md
├── requirements.txt
│
├── notebook/
│   └── street_works_compliance.ipynb
│
├── sample_data/
│   └── input_video.mp4
│
├── sample_output/
│   ├── run_1/
│   └── run_2/
```


## Running the Notebook

### **Environment**

This notebook is designed to run in Google Colab (GPU runtime recommended).
Colab-specific utilities are used for:
* Authentication
* File upload
* Interactive display

### **Steps**

1. Open the notebook in Google Colab.
2. Set runtime to GPU.
3. Install dependencies (first cell).
4. Upload or use provided sample video.
5. Run inference and generate outputs.

### **Dependencies**
See *requirements.txt*

Core libraries include:

* transformers (>= 4.57.0)
* torch
* accelerate
* opencv-python
* pillow
* numpy
* matplotlib
* huggingface_hub

### **Reproducibility Notes**

* Inference outputs may vary slightly across runs due to decoding dynamics.
* Timestamp merging ensures stable reporting behavior.
* Two sample runs are included to illustrate output granularity differences.

## **Conclusion**

Within a narrowly scoped barrier continuity scenario, this project demonstrates that reasoning-oriented visual models can support structured, inspection-style tasks that extend beyond conventional object detection.

The results suggest that:
* Prompt engineering functions as lightweight domain encoding.
* Spatiotemporal reasoning can be operationalized without fine-tuning.
* Modular rule-based reasoning passes may provide a scalable compliance architecture.

This work is intended as a focused exploration of reasoning capability, not a complete regulatory compliance system.