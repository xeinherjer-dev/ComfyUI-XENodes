# Third-Party Licenses and Notices

ComfyUI-XENodes incorporates and adapts model architectures from several open-source inverse tone mapping (SDR to HDR) research projects. This document provides notice of the original authors, projects, and license terms.

---

## 1. HDRTVNet & HDRTVNet++

- **Original Repositories**:
  - HDRTVNet++: [https://github.com/xiaom233/HDRTVNet-plus](https://github.com/xiaom233/HDRTVNet-plus)
  - HDRTVNet: [https://github.com/chxy95/HDRTVNet](https://github.com/chxy95/HDRTVNet)
- **Papers**:
  - *Towards Efficient SDRTV-to-HDRTV by Learning from Image Formation* (TPAMI 2023 / arXiv:2309.04084)
  - *A New Journey From SDRTV to HDRTV* (ICCV 2021)
- **Authors**: Xiangyu Chen, Zheyuan Li, Zhengwen Zhang, Jimmy S. Ren, Yihao Liu, Jingwen He, Yu Qiao, Jiantao Zhou, Chao Dong
- **License**: MIT License
- **Models using this architecture**:
  - `HR.pth` (Highlight Refinement)
  - `AGCM.pth`
  - `LE.pth`

### HDRTVNet MIT License Text

```text
MIT License

Copyright (c) 2023 Hoven Li
Copyright (c) 2021 XyChen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. HDRTVDM

- **Original Repository**: [https://github.com/AndreGuo/HDRTVDM](https://github.com/AndreGuo/HDRTVDM)
- **Paper**: *Learning a Practical SDR-to-HDRTV Up-Conversion Using New Dataset and Degradation Models* (CVPR 2023)
- **Authors**: Cheng Guo, Leidong Fan, Ziyu Xue, Xiuhua Jiang
- **License**: Mozilla Public License Version 2.0 (MPL-2.0)
- **Models using this architecture**:
  - `HDRTVDM_params.pth`
  - `HDRTVDM_params_3DM.pth`
  - `HDRTVDM_params_DaVinci.pth`

The neural network implementation (LSN / Transformer-UNet) adapted from HDRTVDM is governed by the Mozilla Public License 2.0 (MPL-2.0). Components derived from Restormer within HDRTVDM are also governed by the MIT License (see Section 4). Under Section 3.3 of MPL-2.0, this code is combined as part of a Larger Work licensed under GNU General Public License v3.0 (GPL-3.0). A copy of the MPL-2.0 license is available at [https://www.mozilla.org/MPL/2.0/](https://www.mozilla.org/MPL/2.0/).

---

## 3. ITMLUT

- **Original Repository**: [https://github.com/AndreGuo/ITMLUT](https://github.com/AndreGuo/ITMLUT)
- **Paper**: *Redistributing the Precision and Content in 3D-LUT-based Inverse Tone-mapping for HDR/WCG Display* (ACM SIGGRAPH CVMP 2023)
- **Authors**: Cheng Guo, Leidong Fan, Qian Zhang, Hanyuan Liu, Kanglin Liu, Xiuhua Jiang
- **License**: Mozilla Public License Version 2.0 (MPL-2.0)
- **Models using this architecture**:
  - `ITMLUT_params.pth`
  - `ITMLUT_params_TV1K.pth`

The 3D-LUT based inverse tone mapping implementation adapted from ITMLUT is governed by the Mozilla Public License 2.0 (MPL-2.0). The adaptive vertex interpolation algorithm (`pure_ailut_transform`) is adapted from the AiLUT operator in AdaInt under Apache License 2.0 (see Section 5). Under Section 3.3 of MPL-2.0, this code is combined as part of a Larger Work licensed under GNU General Public License v3.0 (GPL-3.0). A copy of the MPL-2.0 license is available at [https://www.mozilla.org/MPL/2.0/](https://www.mozilla.org/MPL/2.0/).

---

## 4. Restormer

- **Original Repository**: [https://github.com/swz30/Restormer](https://github.com/swz30/Restormer)
- **Paper**: *Restormer: Efficient Transformer for High-Resolution Image Restoration* (CVPR 2022)
- **Authors**: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang
- **License**: MIT License
- **Usage**: Used in HDRTVDM Transformer-UNet blocks (`RestormerAttention`, `RestormerTransformerBlock`, `RestormerLayerNorm`, `BiasFree_LayerNorm`).

### Restormer MIT License Text

```text
MIT License

Copyright (c) 2022 Syed Waqas Zamir and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 5. AdaInt

- **Original Repository**: [https://github.com/ImCharlesY/AdaInt](https://github.com/ImCharlesY/AdaInt)
- **Paper**: *AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-Time Image Enhancement* (CVPR 2022)
- **Authors**: Canqian Yang, Meiguang Jin, Xu Jia, Yi Xu, Ying Chen
- **License**: Apache License 2.0
- **Usage**: The trilinear interpolation over non-uniform 3D LUT vertices in `pure_ailut_transform` is a pure PyTorch re-implementation of the AiLUT transform operator from AdaInt.

A copy of the Apache License 2.0 is available at [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Notice Regarding Pretrained Weights & Training Datasets

- **Distribution Method**: Model weight checkpoint files (`.pth`) are not bundled or redistributed within this repository. Instead, they are downloaded on-demand directly from their original public repositories by the end-user's local environment.
- **Dataset / Non-Commercial Constraints**: Some pretrained models (including HDRTVNet/HDRTVNet++ checkpoints) were trained on datasets (such as HDRTV1K) curated from public video sources with an explicit academic use expectation (*"Please download this dataset only for academic use"*). Users intending to deploy these models in commercial products or pipelines should review the underlying datasets and licensing terms of the original authors.
