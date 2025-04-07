# Fine-grained-action-recognition-with-InternVideo2
This project extends InternVideo2, a general-purpose vision foundation model, by introducing a Region-of-Interest (ROI) Alignment Module and proposing a branch-based training strategy for feature fusion. These innovations significantly enhance multimodal foundation models' capability in comprehending fine-grained information.

## Architectural Improvements
- **ROI Alignment Module**: Enables precise spatial correspondence between visual and textual modalities
- **Branch-based Feature Fusion**: Implements parallel processing streams with dynamic gating mechanism

## Key Innovations
| Module | Functionality | Performance Gain |
|--------|---------------|-------------------|
| ROI Alignment | Cross-modal spatial correlation mapping | -|
| Branch Fusion | Multiscale feature integration | Acc 73.12% on ssv2|

## Reference
```bibtex
@article{internvideo2,
  title={InternVideo2: Scaling Video Foundation Models for Multimodal Understanding},
  author={Authors},
  journal={arXiv},
  year={2023}
}
