# 工作记录 —— Paper & News
## 01/07/2024-07/07/2024
**Automatic 3D+t four-chamber CMR quantification of the UK biobank: 
integrating imaging and non-imaging data priors at scale**

MCSI-Net (Multi-Cue Shape Inference Network), where we embed a statistical shape model 
inside a convolutional neural network and leverage both phenotypic and demographic information
from the cohort to infer subject-specific reconstructions of all four cardiac chambers in 3D. 
In this way, we leverage the ability of the network to learn the appearance of cardiac chambers
in cine cardiac magnetic resonance (CMR) imag

![image](/images/3dt_network.png)

> Deep learning with traditional algorithms 

> 通过六维数据的辅助，将统计模型嵌入深度学习模型生成心脏形状
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

> Pipeline for CMR图像分割以及分析
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

> 3D reconstruction + point clouds

> 提出了一种不同于传统迭代推理得到结果的方法，提出了一种基于深度学习的网络替代迭代，而是直接通过稀疏结果预测，
> 从稀疏的CMR图像出发，通过点云重构心脏mesh

## 08/07/2024-14/07/2024
**Recovering from Missing Data in Population Imaging – Cardiac MR Image Imputation via
Conditional Generative Adversarial Nets**

In this work, we propose a new robust approach, coined Image Imputation
Generative Adversarial Network (I2-GAN), to learn key features of cardiac short axis
(SAX) slices near missing information, and use them as conditional variables to infer
missing slices in the query volumes. In I2-GAN, the slices are first mapped to latent
vectors with position features through a regression net. The latent vector corresponding
to the desired position is then projected onto the slice manifold, conditioned on intensity
features through a generator net. The generator comprises residual blocks with normalisation layers that are modulated with auxiliary slice information, enabling propagation
of fine details through the network. In addition, a multi-scale discriminator was implemented, along with a discriminator-based feature matching loss, to further enhance
performance and encourage the synthesis of visually realistic slices。

![image](/images/cardiac_conditional_gan.png)
![image](/images/cardiac_generator.png)

> 3D conv + GAN 

> 通过CMR的残缺图像，复原原本的CMR投影

---
**Image-derived phenotype extraction for genetic
discovery via unsupervised deep learning in
CMR images**

Therefore, the latent
variables produced by the encoder condense the information related to
the geometry of the biologic structure of interest. The network’s training proceeds 
in two steps: the first is genotype-agnostic and the second
enforces an association with a set of genetic markers selected via GWAS
on the intermediate latent representation. This genotype-dependent optimisation procedure 
allows the refinement of the phenotypes produced
by the autoencoder to better understand the effect of the genetic markers encountered. 
We tested and validated our proposed method on leftventricular meshes derived from cardiovascular magnetic resonance images from the UKB, leading to the discovery of novel genetic associations
that, to the best of our knowledge, had not been yet reported in the literature on cardiac phenotypes.

![image](/images/gene_discovery_unsupervised.png)

> 图卷积网络，GWAS,自编码器

> 通过自动编码器在图像衍生的三维网格上操作，以无监督的方式进行表型分析，发现有关心脏表型新的遗传关联

> 该方向对我来说稍微有点困难

---
**Predicting Myocardial Infarction through Retinal Scans and Minimal
Personal Information**

We
trained a multi-channel variational autoencoder (mcVAE) and a deep regressor model to estimate LVM (4.4
(-32.30, 41.1) g) and LVEDV (3.02 (-53.45, 59.49) ml) and predict risk of myocardial infarction (AUC=0.80±
0.02, Sensitivity=0.74 ± 0.02, Specificity=0.71 ± 0.03) using just the retinal images and demographic data.

> 仅通过视网膜图像与人口统计学数据估算LVM与LVEDV 与 用mcVAE以及 deep regression model来训练
> 预测心肌梗死的风险

> Used the mcVAE trained on all the 5,663 retinal images available of size 128×128px. 

![image](/images/disease_prediction_mcVAE.png)

---
**3D Cardiac Shape Prediction with Deep Neural
Networks: Simultaneous Use of Images and
Patient Metadata**

To the best
of our knowledge, this is the first work that uses such an approach for
3D cardiac shape prediction. We validated our proposed CMR analytics method against a reference cohort containing 500 3D shapes of the
cardiac ventricles. Our results show broadly significant agreement with
the reference shapes in terms of the estimated volume of the cardiac
ventricles, myocardial mass, 3D Dice, and mean and Hausdorff distance.

![image](/images/3d_prediction_cardiac_shape.png)

> 第一篇，使用深度学习模型预测心脏的形状