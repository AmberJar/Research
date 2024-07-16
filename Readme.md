# 工作记录 —— Paper & News
<details>
<summary>01/07/2024-07/07/2024</summary>

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
</details>

<details>
<summary>08/07/2024-14/07/2024</summary>

## 08/07/2024-14/07/2024

**Recovering from Missing Data in Population Imaging – Cardiac MR Image Imputation via Conditional Generative Adversarial Nets**

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

> 第一篇使用深度学习模型预测心脏的形状


---
SNN 脉冲神经网络
![image](/images/snn.png)

与 ANN 不同的是，SNN 使用脉冲的序列来传递信息，每个脉冲神经元都经历着丰富的动态行为。
具体而言，除了空间域中的信息传播外，时间域中的过去历史也会对当前状态产生紧密的影响。
因此，与主要通过空间传播和连续激活的神经网络相比，神经网络通常具有更多的时间通用性，但精度较低。
由于只有当膜电位超过一个阈值时才会激发尖峰信号，因此整个尖峰信号通常很稀疏。
此外，由于尖峰值 (Spike) 是二进制的，即0或1，如果积分时间窗口  调整为1，输入和权重之间的乘法运算就可以消除。
由于上述原因，与计算量较大的 ANN 网络相比，SNN 网络通常可以获得较低的功耗。

---
**Agent Attention: On the Integration of Softmax and Linear Attention**
集成Softmax和Linear注意力机制
![image](/images/agent_attention.png)

___
**A Multimodal, Multi-Task Adapting Framework for Video Action Recognition**
综上所述，我们的贡献有三点：1）我们提出了一种新颖的多模态、多任务适配框架，将强大的CLIP模型转移到视频动作识别任务中。
该方法在确保最先进的零样本可转移性的同时，实现了强大的监督性能，如图2所示。
2）我们设计了一种新的视觉TED-Adapter，执行时间增强和差分建模，以增强视频编码器的表示能力。同时，我们为文本编码器引入了适配器，使标签表示可学习和可调节。
3）我们引入了一个多任务解码器，以提高整个框架的学习能力，巧妙地在监督性能和泛化能力之间实现平衡。

___
**An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels**

PiT最大的缺点就在于，移除patch这个单位后会造成输入序列过长，这对Transformer架构而言是一个致命问题——计算成本会随序列长度大幅增加。

___
**A Unified Framework for 3D Scene Understanding**

提出了一个简单且有效的3D点云统一分割框架：UniSeg3D模型。这一模型的设计理念是，构建一个统一的框架同时处理六种3D点云分割任务，通过多任务交互充分挖掘任务间的协同性，以实现全面而深入的场景理解，从而进一步促进3D点云分割任务中的性能表现。UniSeg3D框架有如下的优势:

多任务统一：当前的3D点云分割方法通常为单一任务设计，不同于现有的研究工作，UniSeg3D经过一次推理过程能够同时支持六种点云分割任务；
性能优异：通过建立任务间的显式关联，UniSeg3D在全景分割、语义分割、实例分割、交互式分割、参考分割和开放词汇语义分割六个任务中均展现出SOTA性能；
可扩展性：UniSeg3D采用query统一表征多种点云分割任务的信息与特征，结构简洁有效。且通过输入新增任务的query表征，可将UniSeg3D拓展至更多任务，展现了框架的可扩展性和灵活性。

![image](/images/3d_cloud_unified_framework.png)

___
**Autoregressive Image Generation without Vector Quantization**

将扩散过程中的损失函数引入到自回归图像生成过程，引入了扩散损失（Diffusion Loss）

自回归模型学习不同token间的关联性，而扩散过程通过损失函数学习单个token的概率分布。具体来讲，
自回归模型会根据前面的token预测一个向量z作为小型去噪网络（如MLP）的条件，
通过损失函数不断学习连续值x的潜在分布并从p(x|z)中采样。

![image](/images/autoregression_diffuison_loss.png)

___
**SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization (ICML 2024)**
探索用 BatchNorm 替换 LayerNorm 来加速 Transformer 的推理过程。BatchNorm 导致较低的推理延迟，但可能导致训练崩溃和性能较差，
而 LayerNorm 可以稳定训练，但在推理过程中具有额外的计算成本。为此，本文提出一种渐进策略，通过使用超参数来控制两个层的比例，
将 LayerNorm 逐渐替换为 BatchNorm。作者还提出了一种新的 BatchNorm (RepBN) 重参数化方法，以提高训练稳定性和整体性能。
作者提出了一个简化的线性注意 (Simplified Linear Attention, SLA) 模块，该模块利用 ReLU 作为核函数，并结合深度卷积进行局部特征增强。
所提出的注意力机制比以前的线性注意力更有效，但仍然获得了相当的性能。
本文的渐进式重参数化 BatchNorm 在图像分类和目标检测任务上表现出了强大的性能，以较低的推理延时获得了相当的精度。

![image](/images/SLA.png)

---
**Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer**

扩散模型在图像生成方面表现出了很显著的性能。然而对于生成超高分辨率的图像 (比如 4096 ×4096) 而言，由于其 Memory 也会二次方增加，
因此生成的图像的分辨率通常限制在 1024×1024。在这项工作中。作者提出了一种单向块注意力机制，可以在推理过程中自适应地调整显存开销并处理全局依赖关系。
在这个模块的基础上，作者使用 DiT 的架构，并逐渐执行上采样，最终开发了一个无限的超分辨率模型 Inf-DiT，能够对各种形状和分辨率的图像进行上采样。综合实验表明，
Inf-DiT 在生成超高分辨率图像方面取得了 SOTA 性能。与常用的 UNet 结构相比，Inf-DiT 在生成 4096×4096 图像时可以节省超过5倍显存。

1. 提出了单向块注意力机制 (Unidirectional Block Attention，UniBA) 算法，在推理过程中将最小显存消耗从  降低到 , 其中  表示边长。该机制还能够通过调整并行生成的块数量、在显存和时间开销之间进行权衡来适应各种显存限制。
2. 基于这些方法，训练了一个图像上采样扩散模型 Inf-DiT，这是一个 700M 的模型，能够对不同分辨率的和形状图像进行上采样。Inf-DiT 在机器 (HPDV2 和 DIV2K 数据集) 和人工评估中都实现了最先进的性能。
3. 设计了多种技术来进一步增强局部和全局一致性，并为灵活的文本控制提供 Zero-Shot 的能力。

---
**Guidance with Spherical Gaussian Constraint for Conditional Diffusion**

最近的Guidance方法试图通过利用预训练的扩散模型实现损失函数引导的、无需训练的条件生成。虽然这些方法取得了一定的成功，
但它们通常会损失生成样本的质量，并且只能使用较小的Guidance步长，从而导致较长的采样过程。

在本文中，我们揭示了导致这一现象的原因，即采样过程中的流形偏离（Manifold Deviation）。我们通过建立引导过程中估计误差的下界，从理论上证明了流形偏离的存在。
为了解决这个问题，我们提出了基于球形高斯约束的Guidance方法（DSG），通过解决一个优化问题将Guidance步长约束在中间数据流形内，使得更大的引导步长可以被使用。
此外，我们提出了该DSG的闭式解（Closed-Form Solution）, 仅用几行代码，就能够使得DSG可以无缝地插入(Plug-and-Play)到现有的无需训练的条件扩散方法，
在几乎不产生额外的计算开销的同时大幅改善了模型性能。我们在各个条件生成任务（Inpainting, Super Resolution, Gaussian Deblurring, 
Text-Segmentation Guidance, Style Guidance, Text-Style Guidance, and FaceID Guidance）中验证了DSG的有效性。

---
**Contextual Position Encoding:Learning to Count What’s Important**

总的来说，该研究提出了一种新的用于 transformer 的位置编码方法 CoPE（全称 Contextual Position Encoding），解决了标准 transformer 无法解决的计数和复制任务。
传统的位置编码方法通常基于 token 位置，而 CoPE 允许模型根据内容和上下文来选择性地编码位置。CoPE 使得模型能更好地处理需要对输入数据结构和语义内容进行精细理解的任务。
文章通过多个实验展示了 CoPE 在处理选择性复制、计数任务以及语言和编码任务中相对于传统方法的优越性，尤其是在处理分布外数据和需要高泛化能力的任务上表现出更强的性能。
CoPE 为大型语言模型提供了一种更为高效和灵活的位置编码方式，拓宽了模型在自然语言处理领域的应用范围。
有网友表示，CoPE 的出现改变了在 LLM 中进行位置编码的游戏规则，此后，研究者能够在一个句子中精确定位特定的单词、名词或句子，这一研究非常令人兴奋。
</details>


<details>
<summary>08/07/2024-14/07/2024</summary>

---
## 15/07/2024-21/07/2024
### VAE及其发展
#### AE
编解码结构。Encoder网络类似于我们使用主成分分析（PCA）或矩阵分解（MF）进行数据降维的过程。
此外，Autoencoder对于从隐编码中恢复数据的过程进行了特别的优化。一个良好的中间表示（即隐编码）
不仅能够有效捕捉数据的潜在变量，
还对数据解压缩过程有所助益。
#### Denoising AE
Denoising Autoencoder是Autoencoder的一个变体，专门用于数据去噪和更鲁棒的特征学习。它通过在输入数据中引入噪声，然后训练网络恢复原始未受扰动的数据。这个过程迫使网络学习更为鲁棒的数据表示，忽略随机噪声，从而提高模型对输入数据中噪声或缺失值的容忍度。

由于Autoencoder（自编码器）学习的是恒等函数, 当网络的参数数量超过数据本身的复杂度时, 
存在过拟合的风险。为了避免这一问题并提高模型的鲁棒性, 这种改进方法是通过在输入向量中加入随机噪声或遮盖某些值来扰动输入, 
即 x* ~ Corruptioin(x), 然后使用 x* 训练模型恢复出原始未扰动的输入 x 。
#### Sparse AE
Sparse Autoencoder（稀疏自编码器）是自编码器的一个变体，它在隐藏层上应用“稀疏”约束以防止过拟合并增强模型的鲁棒性。
该模型通过限制隐藏层中同时激活的神经元数量，强制使大部分神经元大多数时间处于非激活状态。
#### Contractive AE
Contractive Autoencoder（收缩自编码器）是一种自编码器，旨在通过学习鲁棒性更高的数据表示来提高模型性能。与K-Sparse Autoencoder类似，
它通过在损失函数中加入额外的项来鼓励模型在被压缩的空间中学习更稳健的表示。
#### VAE
详见笔记
#### Conditional VAE
Conditional Variational Autoencoder（CVAE）是Variational Autoencoder（VAE）的一种扩展，它通过引入额外的条件变量来控制生成过程。
CVAE能够根据给定的条件信息生成特定类型的数据，使得生成的数据不仅多样化且更具针对性
#### Beta-VAE
Beta-VAE（β-VAE）是Variational Autoencoder（VAE）的一个变体，由Higgins et al. 在2017年提出。
其核心目标是发现解耦或分解的潜在因子。解耦表示具有良好的可解释性，并且易于泛化到多种任务。例如，在人脸图像生成中，
模型可能分别捕捉到微笑、肤色、发色、发长、情绪、是否佩戴眼镜等相对独立的因素。
#### VQ-VAE
VQ-VAE（Vector Quantised-Variational AutoEncoder），由van den Oord等人于2017年提出，
是一种结合了变分自编码器和向量量化技术的模型。这种模型特别适用于处理自然语言处理、语音识别等任务，
因为这些任务中的数据往往更适合用离散而非连续的表示。VQ-VAE的核心创新在于其潜在空间的离散化，
这种离散化使得模型能够有效地处理和生成高度结构化的数据。 与传统的VAE相比，VQ-VAE在处理某些类型的数据时更为自然和高效，
特别是在需要将输入数据映射到有限的离散空间时。
这种离散的表示方式为复杂数据模式的学习提供了新的可能性。
#### VQ-VAE-2
VQ-VAE-2，由Ali Razavi等人于2019年提出，是VQ-VAE的升级版。它引入了一个层次化的结构，
旨在更细致地捕捉数据中的局部和全局信息。通过这种层次化设计，VQ-VAE-2能够更有效地捕获数据的多尺度特性，
从而生成更高质量的图像。 这个模型的一个显著特点是，它结合了自回归模型和自注意力机制，
进一步增强了其在复杂数据生成任务中的性能。VQ-VAE-2在图像生成领域表现出了特别的实力，尤其是在细节表现和多样性上，展现了显著的提升。

---
### FreeU: Free Lunch in Diffusion U-Net
低频分量表现出了较低的变化率，但是高频分量在整个的去噪过程中表现出了更加明显的变化。这些发现进一步在图3中得到了证实。
![image](/images/free_unet_1.png)

1. 低频分量体现了图像的全局结构，包括全局的布局和平滑的颜色。这些全局的元素构成了图像的本质。
对于去噪的过程，这些快速变化其实是不合理的。因此，低频分量的快速变化会重塑图像的本质，这也与去噪的目标是矛盾的。
2. 高频分量包含图像的边缘和纹理信息。这些精细的细节对噪声非常敏感。而噪声被引入图像时，
通常表现为随机的高频信息。因此，去噪过程需要在擦除噪声的前提下，同时保持重要且复杂的细节。

![image](/images/free_unet_2.png)
![image](/images/free_unet_3.png)

___
### Vlogger: Make Your Dream A Vlog
With such a design of mimicking human beings, our
Vlogger can generate vlogs through explainable cooperation of top-down planning and bottom-up shooting. Moreover, we introduce a novel video diffusion model, ShowMaker, which serves as a videographer in our Vlogger for
generating the video snippet of each shooting scene. By
incorporating Script and Actor attentively as textual and
visual prompts, it can effectively enhance spatial-temporal
coherence in the snippet. Besides, we design a concise
mixed training paradigm for ShowMaker, boosting its capacity for both T2V generation and prediction. Finally, the
extensive experiments show that our method achieves stateof-the-art performance on zero-shot T2V generation and
prediction tasks. 
![image](/images/vlogger.png)
![image](/images/showmaker.png)

___
### VBench: Comprehensive Benchmark Suite for Video Generative Models
视频生成评价套件

![image](/images/vbench.png)

考虑用于设计损失函数

---
### VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models


</details>