# 重构后代码与实验记录（Release Bundle）

## 一、目录
```
release/code/archs/     网络结构（本文方法的全部模块）
  irlite_variants_arch.py    主干：可折叠重参数化块 BlockRFDXRep、BlockRFDXDW、
                             退化感知 BlockDegMoE、多尺度 BlockMSRDB、IRLiteVariant 主体
  crossmodal_blocks.py       可见光引导：SGHFusion（低频取红外 + 唯一高频注入 + 判据门控）
  deg_sem_modules.py         退化/语义模块（CGHFv2 低分辨率注意力、WideMoE 分组、VQ 码本）
  guided_baselines.py        文献式引导基线（8 种，用于同协议对照）
release/code/tools/     语义-物理管线（下一阶段 PhysIR / OVPP-Net）
  run_sources.py            多源语义估计（S1 EVF-SAM / S2 CLIPSeg / S3 Qwen-VL / S4 SigLIP
                            / S5 跨模态一致性 / S6 时序一致性），含 --mock 模式
  vote_labels.py            加权投票 / 弃权 / 平滑 / 级联 / 可靠性标定
  soft_topk.py              top-k 候选集 + 物理软加权 + 双条件弃权
  phys_table.py             物理知识表加载（发射率区间等）
  physics_veto.py           z 检验物理否决
  build_prior_fields.py     7 通道物理先验场构建
  redegrade.py              可微重退化（模糊→噪声→量化→辐射→大气）
release/code/eval/      评测与审计
  eval_m3fd_uni.py          统一协议串行评测（读各 run 的 yml 重建同构模型后评测）
  audit_report.py           报告一致性审计（图/表/公式/小节编号、引用双向、题注禁用词）
  audit_figures.py          图表尺寸与题注审计（栏宽 15.5 cm、印高 ≤7 cm）
  check_progress.py         训练进度检查（只认最新日志、千分位修正）
  redraw_radar.py           多准则横版对比图（1×4 面板）
release/code/options/   关键配置（本文标准配置、引导协议配置、统一批次配置）
release/code/*.yaml     物理知识表（发射率/光谱行为/否决规则/大气先验/弃权策略）
release/code/train*.py  单进程多模型共享 dataloader 的训练入口
```

## 二、环境
Python 3.8（`repfadn` 环境）、PyTorch + CUDA、BASICSR 框架；`python -m basicsr.train -opt <yml>` 方式启动（必须用 `-m`，直接跑脚本不会注册自定义模块）。

## 三、关键可复现命令
```bash
# 训练（本文标准配置 / 引导协议配置）
python -m basicsr.train -opt release/code/options/train_irlite_sghf_m3fd.yml

# 统一协议评测（串行，避免多进程互相截断）
python release/code/eval/eval_m3fd_uni.py

# 报告一致性审计
python release/code/eval/audit_report.py
python release/code/eval/audit_figures.py
```

## 四、实验记录
- `docs/advisor/_internal/实验索引.csv`：全部实验（名称 / 最佳 PSNR / SSIM / 迭代 / 日志份数）
- `docs/advisor/_internal/eval_logs/`：各评测脚本的原始输出（表 6/9/11/12 的数值出处）
- `docs/advisor/_internal/M3FD_UNI_结果_内部记录.md`：统一批次（从零训练）结果，未进入报告

## 五、已知边界
- 报告表 13 中 † 标记的两行取自"60 k 权重微调 40 k"批次，与本表其余行批次不同（表下已注明）；
- 引导增益的可复现性与训练预算强相关：同预算（从零 40 k）下引导增益不显著，详见 _internal 记录。

## 六、权重上传说明
本目录只放**报告表格实际引用**的实验的最终 checkpoint（每个实验 1 个），体积约数 MB 级；
`experiments/` 下的全量 4414 个 checkpoint（约 81 GB，含中间快照与大规模对比实验）不随本包上传。
基础模型（CLIPSeg、SAM2、Qwen2.5-VL、SigLIP、Grounding-DINO 等，合计约 5.6 GB）请从原始发布渠道获取，不随本包分发。

上传命令示例（密钥 `~/.ssh/id_ed25519_release` 已生成）：
```bash
# 代码 + 最小权重集（约 20 MB）
rsync -avz -e "ssh -i ~/.ssh/id_ed25519_release" release/ user@host:/path/to/fdrnet_release/
# 或走 git（代码部分）
cd release && git init && git add . && git commit -m "FDR-Net release"
git remote add origin git@github-fdrnet:<user>/fdrnet-release.git && git push -u origin main
```
