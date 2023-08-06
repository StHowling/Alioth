# Alioth (IPDPS 2023)

Alioth: A Machine Learning Based Interference-Aware Performance Monitor for Multi-Tenancy Applications in Public Cloud

Multi-tenancy in public clouds may lead to co-location interference on shared resources, which possibly results in performance degradation of cloud applications. However, due to black-box nature of VMs in the IaaS public cloud, cloud providers cannot acquire application-level performance information and mainly rely on low-level metrics such as CPU usage and hardware counters, to determine when such events happen and how serious the degradation is. We propose Alioth, a machine learning framework to monitor the performance degradation of cloud applications based on low-level metrics [[arXiv]](https://arxiv.org/abs/2307.08949) [[IEEE Xplore]](https://ieeexplore.ieee.org/abstract/document/10177453).

Although ML approaches to identify co-location interference appear plausible, they suffer from the data-hungry nature and lacking of interpretability. To feed the models, we first elaborated interference generators and profiled some cloud applications in the lab environment to build *Alioth-dataset*. **The dataset is also open-sourced in this repository**. We hope the releasement can help people get a better understanding of this problem and benefit future research.

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{shi2023alioth,
  title={Alioth: A Machine Learning Based Interference-Aware Performance Monitor for Multi-Tenancy Applications in Public Cloud},
  author={Tianyao Shi and Yingxuan Yang and Yunlong Cheng and Xiaofeng Gao and Zhen Fang and Yongqiang Yang},
  booktitle={IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  year={2023},
  pages={908--917}
}
```