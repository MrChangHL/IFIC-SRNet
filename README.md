Blind Super-Resolution Based on Interframe Information Compensation for Satellite Video (IEEE J-STARS 2025)

📖[Paper](https://ieeexplore.ieee.org/document/11130431)🎁[Code](https://github.com/MrChangHL/IFIC-SRNet)
Authors:Hongliang Chang, Haijiang Sun, Jinchang Ren, Qiaoyuan Liu, Xiaowen Zhang

Abstract
Super-Resolution (SR) of satellite video has long been a critical research direction in the field of remote sensing video processing and analysis, and blind SR has attracted increasing attention in the face of satellite video with unknown degradation. However, existing blind SR methods mainly focus on accurate blur kernel estimation, while ignoring the importance of inter-frame infor-mation compensation in the time domain. Therefore, this paper focuses on precise temporal information compensation and pro-poses a blind SR Network based on Inter-Frame Information Compensation (IFIC-SRNet). First, we propose a Multi-Scale Parallel Convolution block (MSPC) to alleviate the difficulty of alignment between satellite video frames due to the presence of moving objects of different scales. Second, we propose a Hybrid Attention-based Feature Extraction Module (HAFEM) that ef-fectively extracts both local and global information between vid-eo frames. While activating more pixels, more attention is allo-cated to informative pixels to obtain the clean features. Finally, a Pyramid Space Activation Module (PSAM) is proposed to gradu-ally adjust the clean features through a multi-layer iterative pyr-amid structure, enabling the clean features to better perceive blur and achieve pixel-level fine compensation for unknown de-graded frames. Extensive experiments on real satellite video da-tasets demonstrate that our method is superior to state-of-the-art non-blind and blind SR methods, both qualitatively and quantita-tively.

Environment
Cuda 11.8
Pytorch 
