# ECNNet(building)

## 📖 Abstract
Colorectal polyp segmentation from colonoscopy images remains challenging due to variable shapes, blurred boundaries, and heterogeneous appearances. Existing methods suffer from insufficient multi-scale semantic modeling and limited use of cross-domain structural cues. Here we present a dual-domain structural guidance framework that integrates grayscale-guided edge features and graph-based multi-scale fusion to improve segmentation robustness. The method employs a learnable edge extractor initialized with Sobel operators to capture complementary high-frequency information from RGB and grayscale domains, and uses graph convolution to model semantic dependencies across multi-scale branches. We also introduce an edge-guided progressive decoder to refine boundary details. Experiments on five public benchmarks show the method achieves 93.62% Dice on Kvasir-SEG and 95.06% Dice on CVC-ClinicDB, outperforming many state-of-the-art approaches, especially for small and flat polyps. This work provides a reliable visual computing solution for clinical endoscopic image analysis.

## 📂 Repository Structure
```text
.
├── utils/
│   └── ASPPGCN.py          # Implementation of Graph-ASPP
├── model.py
├── train.py            
├── requirements.txt        # Dependencies
└── README.md
```


## 🔧 Preparation
```text
git clone https://github.com/ZCX327/ECNNet
pip install -r requirements.txt
```
