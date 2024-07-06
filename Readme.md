# 工作记录 —— Paper & News
## 06/07/2024
**Automatic 3D+t four-chamber CMR quantification of the UK biobank: 
integrating imaging and non-imaging data priors at scale**

MCSI-Net (Multi-Cue Shape Inference Network), where we embed a statistical shape model 
inside a convolutional neural network and leverage both phenotypic and demographic information
from the cohort to infer subject-specific reconstructions of all four cardiac chambers in 3D. 
In this way, we leverage the ability of the network to learn the appearance of cardiac chambers
in cine cardiac magnetic resonance (CMR) imag

![image](/images/3dt_network.png)

> Deep learning with traditional algorithms

---

**Quantitative CMR population imaging on 20,000 subjects of the UK
Biobank imaging study: LV/RV quantification pipeline and its
evaluation**

we present and evaluate a cardiac magnetic resonance (CMR) image analysis pipeline that properly
scales up and can provide a fully automatic analysis of the UKB CMR study. Without manual user interactions, 
our pipeline performs end-to-end image analytics from multi-view cine CMR images all the way to anatomical and 
functional bi-ventricular quantification. All this, while maintaining relevant quality controls of the CMR input images, 
and resulting image segmentations.

> Traditional algorithms and machine learning

---
**Shape registration with learned deformations for 3D shape
reconstruction from sparse and incomplete point clouds**

MR-Net enables
accurate 3D mesh reconstruction in real-time despite missing data and with sparse annotations. Using 3D
cardiac shape reconstruction from 2D contours defined on short-axis cardiac magnetic resonance image
slices as an exemplar, we demonstrate that our approach consistently outperforms state-of-the-art techniques for shape 
reconstruction from unstructured point clouds. 

Traditional 3D shape reconstruction approaches have relied on
iterative deformation of a template mesh to sparse contours/PC,
using the latter to guide the former, with including various penalty
terms to ensure the estimated deformation is smooth. To eliminate
the requirement of several iterations during inference (which can
be time-consuming), in this paper, a deep learning-based network,
MR-Net, is designed to mimic such a process.

MR-Net is to reconstruct personalised meshes from sparse contours under the guidance of a template mesh

![image](/images/3d_reconstruction_regis.png)