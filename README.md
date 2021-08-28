# ICCV2021_Event_Frame_Tracking

## Object Tracking by Jointly Exploiting Frame and Event Domain (ICCV 2021)

Jiqing Zhang, Xin Yang, Yingkai Fu, Xiaopeng Wei, Baocai Yin, Bo Dong

[[project](https://zhangjiqing.com/publication/object-tracking-by-jointly-exploiting-frame-and-event-domain-iccv-2021/)] [paper] [[dataset](https://zhangjiqing.com/dataset/)]

## Abstract

Inspired by the complementarity between conventional frame-based and bio-inspired event-based cameras, we propose a multi-modal based approach to fuse visual cues from the frame- and event-domain to enhance the single object tracking performance, especially in degraded conditions (e.g., scenes with high dynamic range, low light, and fast motion objects). The proposed approach can effectively and adaptively combine meaningful information from both domains. Our approach’s effectiveness is enforced by a novel designed cross-domain attention schemes, which can effectively enhance features based on self- and cross-domain attention schemes; The adaptiveness is guarded by a specially designed weighting scheme, which can adaptively balance the contribution of the two domains. To exploit event-based visual cues in single-object tracking, we construct a largescale frame-event-based dataset, which we subsequently employ to train a novel frame-event fusion based model. Extensive experiments show that the proposed approach outperforms state-of-the-art frame-based tracking methods by at least 10.4% and 11.9% in terms of representative success rate and precision rate, respectively. Besides, the effectiveness of each key component of our approach is evidenced by our thorough ablation study.

## Requirements
\cdot CUDA == 11.1
\cdot Pytorch == '1.7.0+cu110'
