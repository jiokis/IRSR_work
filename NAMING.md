# 正式命名对照表（论文用）

论文与代码中的称谓一律使用下表左列的正式术语；右列的内部代号（历史遗留）不再出现在正文、图题、表头与对外交付物中。

## 一、方法命名
| 正式称谓 | 说明 | 内部代号（不再使用） |
| --- | --- | --- |
| **本文方法** | 全文统一称"本文方法"；给出一次全称后不再重复 | irlite / IRLite / RepMDSR / RepFADN |
| **可折叠重参数化红外超分网络** | 首次出现时的全称（暂定，可替换为下面的备选名） | RepMDSR |

### 方法名备选（择一）
1. **FDR-Net**：Foldable–Degradation-aware–Reliability-gated Network，三处创新各取一词（推荐，缩写信息量最高）
2. **FRDA-IRSR**：Foldable Reparameterized Degradation-Aware Infrared SR（描述最完整）
3. **保形重参数化红外超分网络**（中文全称，英文缩写 SH-RN，语感偏工程）——不推荐

## 二、模块命名
| 正式称谓 | 内部代号 |
| --- | --- |
| 再循环特征蒸馏块 | rfdx / BlockRFDX |
| **可折叠多分支重参数化块** | rfdxrep / rfdxrepe / BlockRFDXRep / BlockRFDXRepEdge |
| 深度可分离可折叠块 | BlockRFDXDW |
| 多尺度密集块 | msrdb / BlockMSRDB |
| 退化感知门控专家块 | degMoE / BlockDegMoE |
| 双统计门 | dual-stat gate |
| 结构一致性判据门控的高频融合模块 | sghf / SGHF / SGHFusion |
| 热结构—热强度双路引导 | tscf / TS-CF |
| 低分辨率注意力融合模块 | CGHFv2 |
| 分组专家权重模块 | WideMoE |

## 三、引导对照方法（表 13 / 表 14）
| 正式称谓 | 内部代号 |
| --- | --- |
| 特征拼接＋1×1 融合 | concat1x1 |
| 逐级通道注意力融合 | ca_msff（MSFF-CT 式）|
| 由粗到细渐进引导 | coarse2fine（CPGNet 式）|
| 固定强度高频注入（无判据）| nogate |
| 文献式交叉注意力融合（CAF）| caf_gate / cmf='caf' |
| 免对齐空间注意力融合 | spatattn（AlignFreeNet 式）|
| 相关匹配与变形对齐 | corrmatch（MASA-SR 式）|
| 双向互惠门控 | bidir（RRSR 式）|

## 四、数据集与平台（保持原样，属专有名词）
FLIR-IISR、M3FD、LLVIP、DIV2K、PyTorch、CUDA、BasicSR、ONNX、RKNN、RTX 4070。

## 五、文件与路径命名
| 现状 | 正式命名 |
| --- | --- |
| `fig18_sghf.png` | `fig18_structure_gated_hf.png`（已改 ✓）|
| `fig25_radar.png` | `fig25_multicriteria_bar.png`（建议后续改）|
| `*_sghf_*.yml`、`*_caf_*.yml` 等配置名 | 保留（仅工程内部使用，不进论文）|
