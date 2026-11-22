"""
model_core/config.py — 模型层配置

仅保留模型训练所需的参数。
品种、数据、风控等全局配置统一由根目录 config.py 的 Config 类管理。
"""
import math

import torch
from .vocab import FORMULA_VOCAB


class ModelConfig:
    # ── 训练设备 ─────────────────────────────────────────────────────────
    # 注意：本任务 CPU 训练速度反而比 GPU 快（实测约 2.3 倍），故强制用 CPU。
    # 原因：
    #   1. 张量太小——forex 组仅 (2 品种 × 3508 × 20 特征)，单个算子的
    #      计算量小于 CUDA kernel 启动开销（数十微秒），GPU 算得快但启动慢。
    #   2. 训练循环是 Python 串行调度：每 step 逐条跑 128 条公式 × 8 个
    #      VM 步 × 4 个 walk-forward 折，GPU 被切成上万个碎片段，吃不满。
    #   3. host↔device 拷贝 + kernel 启动延迟主导总耗时，而非张量计算本身。
    #   4. 实测 GPU 利用率 ~51%，正是 GPU 一半时间在干等 Python 喂下一个
    #      kernel 的证据（不是“还能压榨”，而是“调度瓶颈”）。
    # 基准测试（forex 组, 50 步, RTX 4060, 2026-07-03）：
    #   cuda: 4.48 s/步  Best=4.875
    #   cpu : 1.91 s/步  Best=5.103
    #   加速比 = 0.43x（GPU 反而慢 2.3 倍）
    # 若后续改为批量并行公式评估（一次喂大批张量进 GPU），再切回 cuda。
    DEVICE = torch.device("cpu")

    # ── 训练参数（大搜索空间适配版，2026-07-04 重构）─────────────────────
    # 背景：特征库扩展到 65、算子库扩展到 66（vocab=131），8-token 搜索空间
    #   从旧版 ~7亿 暴增到 ~8.67×10^16（1.2 亿倍）。旧的采样预算（128×3000）
    #   覆盖率趋近于零，导致熵坍塌 Early Stop、公式退化。
    # 对策（训练时间不敏感场景）：
    #   1. 特征剪枝（active_features.json）把 vocab 降到 ~90，空间缩小约 20 倍
    #   2. 放大采样预算：BATCH_SIZE 128→256，TRAIN_STEPS 3000→8000
    #   3. 更大精英池（60）保留更多历史最优
    BATCH_SIZE      = 192   # 每步采样公式数（原 128，1.5x 提升覆盖率）
    TRAIN_STEPS     = 9000  # 每组训练步数（55次重启需要更多步数）
    MAX_FORMULA_LEN = 8     # 公式长度上限：保持 8（10 会导致 CPU 训练慢 3 倍）

    # ── 特征维度（由 vocab.py 自动派生，无需手动修改）──────────────────
    INPUT_DIM: int = FORMULA_VOCAB.feature_count  # == 10

    # ── Reward：Sortino 为主，IC 做门控 ──────────────────────────────────
    # IC_NEG_MULT 0.30→0.50：0.30 对反向因子惩罚过重，可能误杀非线性高收益因子。
    # 收益优先模式下，只要年化收益是正的，适当负 IC 可以接受。
    REWARD_ALPHA:      float = 1.0
    IC_GATE_THRESH:    float = 0.01
    IC_GATE_MULT:      float = 1.15
    IC_NEG_MULT:       float = 0.75   # 收益优先：不过度误杀反向/非线性高收益因子

    # ── FTMO 专属奖励模式 ─────────────────────────────────────────────
    # "standard": 收益+风险平衡（默认，原权重）
    # "ftmo":     FTMO 考试盘专属——年化收益权重 0.60→0.75，Calmar 0.05→0.10
    #             （控制 MDD 贴近 10% Max Loss 上限），其余指标权重下调。
    #             目标：在 10% Max Loss 约束下最大化年化收益，快速达标。
    # "forex":    外汇均值回归专属（2026-07-08）——
    #             降年化收益权重(0.80→0.25)、提IC权重(0.03→0.25)、
    #             新增反转奖励(0.20，奖励低/负因子自相关)和多空对称检查(0.15)。
    #             原因：外汇H1以震荡为主，趋势算子效果差，需引导模型偏好
    #             均值回归信号而非追涨杀跌。
    REWARD_MODE:       str = "ftmo"

    # ── 熵保护（大空间加强版）──────────────────────────────────────────
    # ENTROPY_COEFF_MAX 0.5→1.0：加倍探索压力，对抗大 vocab 的过早收敛。
    # ENTROPY_COLLAPSE_THRESH 改为相对阈值 0.15×ln(vocab)：大 vocab 最大熵更高
    #   （ln(131)≈4.87 vs ln(54)≈3.99），绝对阈值 0.5 不再合理。
    # ENTROPY_COLLAPSE_STEPS 15→40：给模型更长的自我恢复窗口，不急于重启。
    ENTROPY_COEFF_MAX:   float = 2.0
    ENTROPY_COEFF_POWER: float = 1.0  # 降低幂次，让低熵时系数更激进（原1.3）
    ENTROPY_COLLAPSE_THRESH: float = 0.15 * math.log(FORMULA_VOCAB.size)
    ENTROPY_COLLAPSE_STEPS:  int   = 20  # 更快检测坍塌并重启

    # ── 熵下限惩罚（Fix 1: H→0 时熵项归零问题）──────────────────────────
    # 当 H < ENTROPY_FLOOR_THRESH 时，加入固定惩罚 λ×(thresh-H)。
    # 这确保即使 mean_ent→0，loss 中仍有非零探索压力。
    ENTROPY_FLOOR:        bool  = True
    ENTROPY_FLOOR_THRESH: float = 1.0   # 熵低于此值时触发固定惩罚（提高介入时机）
    ENTROPY_FLOOR_LAMBDA: float = 5.0   # 惩罚强度系数（加大力度对抗坍塌）

    # ── Elite Replay ──────────────────────────────────────────────────
    ELITE_REPLAY_FRAC:  float = 0.25
    ELITE_POOL_SIZE:    int   = 60    # 30→60：大空间需要更大的精英记忆
    ELITE_REWARD_SCALE: float = 1.2

    # ── 坍塌重启（大空间加强版）─────────────────────────────────────────
    # MAX_RESTARTS 8→25→55、RESTART_NOISE 0.05→0.1→0.25：时间不敏感，多给机会+更强扰动。
    # 配合 engine.py：超过 MAX_RESTARTS 后不再 Early Stop，改为强扰动继续训练。
    # 2026-07-09: US100 训练 24/25 重启仍有突破，扩到 55 次。
    MAX_RESTARTS:   int   = 55
    RESTART_NOISE:  float = 0.25

    # ── 自适应噪声：Best 停滞时自动增大扰动 ─────────────────────────────
    # stagnation_window: 判断停滞的步数窗口
    # noise_min / noise_max: 噪声下界和上界
    # noise_boost: 停滞时噪声提升倍率
    ADAPTIVE_NOISE:      bool  = True
    STAGNATION_WINDOW:   int   = 500
    NOISE_MIN:           float = 0.15
    NOISE_MAX:           float = 0.60
    NOISE_BOOST_FACTOR:  float = 2.0   # noise += 0.2 * (stagnation / window)

    # ── 重启多样性（Fix 2: best_snapshot 吸引子效应）─────────────────────
    # 每 FULL_RESET_EVERY 次重启中，做 1 次完全随机初始化而非从 best_snapshot 恢复。
    FULL_RESET_EVERY:    int   = 3     # 每 3 次重启中第 3 次做 full reset

    # ── Reward baseline（Fix 3: 全负 batch 相对优选问题）──────────────────
    # 用 EMA baseline 替代 batch mean 计算 advantage，避免全负 batch 的问题。
    REWARD_EMA_BASELINE:     bool  = True
    REWARD_EMA_DECAY:        float = 0.95   # EMA 衰减系数
    REWARD_EMA_WARMUP:       int   = 10     # 前 N 步用 batch mean（EMA 未稳定）

    # ── 重启时部分重置参数：保留底层，扰动顶层 ───────────────────────────
    PARTIAL_RESET:       bool  = True
    PARTIAL_RESET_LAYERS: tuple = ("ln_f", "mtp_head", "head_critic", "blocks", "token_emb")

    # ── Elite Replay 衰减：旧 elite 采样权重随时间衰减 ──────────────────
    ELITE_DECAY:         bool  = True
    ELITE_DECAY_HALF_LIFE: int = 300   # 每 300 步旧 elite 权重减半

    # ── 多起点并行（Island）──────────────────────────────────────────────
    # 注意：Island 模式在 CPU 训练下会让总时间变成 N 倍（islands 串行），
    # 对于 index 这类大数组（T=32076）会变得极慢。当前默认关闭，保留配置开关。
    N_ISLANDS:              int   = 1
    MIGRATION_INTERVAL:     int   = 500
    MIGRATION_TOP_K:        int   = 5
    # island 默认关闭，避免用户误开导致速度爆炸

    # ── 因子去相关参数 ────────────────────────────────────────────────
    FACTOR_TOP_K:     int   = 25
    CORR_THRESHOLD:   float = 0.85
    CORR_PENALTY:     float = 0.8

    # ── Walk-Forward Gap ───────────────────────────────────────────────
    # P2-6 修复说明：gap 必须按 target_horizon 标定，且与 CPCV purge gap 统一。
    # 当前 target_horizon=2（data_manager 用 log(open[t+2]/open[t+1]) 作 target_ret），
    # gap=20 相当于 10 个「真实预测步」——对 H1 数据足够，但若切换到 D1/15min
    # 需要重新评估。CPCV 模式（未实现）的 purge gap 应等于 WF_GAP，避免两套阈值。
    # 调整时同时检查：
    #   1. engine.py _build_walk_forward_folds 的 val_start = train_end + gap
    #   2. fold_size 应 >> gap（建议 fold_size >= 5*gap）以保证 val 有效性
    WF_GAP: int = 20

    # ── 公式结构约束（2026-07-05 新增）──────────────────────────────────
    # 背景：index 组因子因 TS_RANK 连续使用退化为 beta 因子（91.8% 做多），
    # 前半段市场跌亏钱、后半段市场涨赚钱，不是 alpha 而是 beta。
    # 对策：在采样阶段禁止恒正算子链，在评分阶段添加 beta 中性 + 前后一致性奖惩。
    ENABLE_FORMULA_STRUCTURE_CONSTRAINT: bool = True   # 总开关
    BETA_NEUTRAL_PENALTY:     bool  = True             # 多空比例失衡惩罚
    HALF_CONSISTENCY_BONUS:   bool  = True             # 前后一致性奖惩
    BETA_NEUTRAL_THRESH:      float = 0.85             # 超过此比例同方向触发重罚
    BETA_NEUTRAL_LIGHT_THRESH: float = 0.70            # 轻度失衡阈值

    # ── 并行评估（2026-11-17 重新开启）──────────────────────────────────
    # 历史：2026-07 曾尝试 ThreadPoolExecutor 并行 Part C，结果反而慢 55%~69%，
    #   原因是每条公式的 Python 编排（遍历 WF 折、.item()、IC、惩罚）全程持
    #   GIL，张量算子太小，收益抵不过线程池 + 超订阅开销，故当时关闭。
    #
    # 现在重新开启的依据——两个关键算子已消除 GIL 阻塞点：
    #   1. ops._ema_simple 由 Python 逐点循环改为 scipy.signal.lfilter
    #      （T=50000 时 2.5 s → 0.6 ms，约 3000x）。lfilter 在 C 层执行且
    #      内部释放 GIL，EMA 链不再霸占解释器锁。
    #      注意 22% 的公式含 EMA，原先单步 1/3 以上时间耗在这里。
    #   2. vm._normalize_output 的滚动标准化由 unfold 改为前缀和差分
    #      （71 ms → 7.6 ms，约 9x），并释放 GIL。
    # 基准复测（XAUUSD M30, T=50000, 96 条公式, i9-13900HX 32 逻辑核）：
    #   串行 intra=32        90.4 ms/公式   ← 旧默认
    #   12w × intra=2        27.7 ms/公式   → 3.3x 加速
    # 维护提示：若今后再引入纯 Python 的逐点循环算子，本组参数需重新基准，
    #   否则可能退回到「并行更慢」的旧状态。串行回退路径复用同一
    #   _eval_formula_task，逻辑与并行完全一致，仅执行方式不同。
    PARALLEL_EVAL:        bool = True
    EVAL_WORKERS:         int  = 12   # 12 物理核（32 逻辑核留余量给 OS/数据加载）
    EVAL_INTRA_THREADS:   int  = 2    # 每 worker 内部 2 线程，12×2=24 < 32 逻辑核

    # ── 主线程（Part A 采样 / Part D 反传）的 intra 线程数 ─────────────────
    # 背景：torch.set_num_threads 是**进程级全局**设置，_init_parallel_eval 里
    #   为公式评估设成 2 之后，Part A 的 Looped Transformer 前向和 Part D 的
    #   REINFORCE 反传也被一并限制到 2 线程——这两个阶段是单线程串行的，
    #   本该吃满所有核。2026-11-17 实测：单步从 4.4 s 退化到 8 s，主线程被
    #   限流是原因之一。
    # 策略：按阶段切换（engine._set_torch_threads），评估阶段用 2 线程避免
    #   12 worker × N 线程超订阅，采样/反传阶段回到满核。
    # 0 = auto（逻辑核数）。
    MAIN_THREADS:         int  = 0

    # ── 续训基准校准（2026-11-17 新增）──────────────────────────────────
    # 问题：续训时 best_score 从旧 checkpoint 恢复。如果期间换了数据文件、
    #   改过奖励权重/折数、或公式解读有变，同一条 best_formula 在当前环境下
    #   可能根本拿不到那个分数——门槛虚高，训练会永远「停滞」在一个不可达
    #   的数字上，看起来在学习其实永远不更新最优。
    #   实测 XAUUSD：记录 1.2141，当前配置复现只有 0.9005，空转 224 步。
    # 对策：训练开始时重新评估一次 best_formula。复现值低于
    #   记录值 × RECALIBRATE_TOLERANCE 时，以复现值为新基准并把停滞计数归零
    #   （保留原公式与权重快照，只是把「要超越的分数线」拉回现实）。
    RECALIBRATE_BEST_ON_RESUME: bool  = True
    RECALIBRATE_TOLERANCE:      float = 0.95   # 复现值 < 记录值×0.95 才校准

    # ── 内存看门狗（2026-11-17 新增）────────────────────────────────────
    # 问题：12 worker 并行评估下，PyTorch CPU 分配器（c10::alloc_cpu →
    #   malloc）在多线程反复分配不同大小张量时会碎片化，进程 RSS 单调上涨
    #   而不回落。实测 26 分钟从 ~3GB 涨到耗尽 31.7GB 物理内存，最后连
    #   663KB 都分配失败，抛出 DefaultCPUAllocator OOM 直接终止训练。
    # 对策：每步读一次自身 RSS，超过 MAX_RSS_GB 就主动干净退出（退出码
    #   MEM_EXIT_CODE），由 train_daemon.py 立刻从最近 checkpoint 续训。
    #   损失上限 = 一个 checkpoint 间隔（默认 20 步），远好过随机 OOM。
    # 设 0 关闭看门狗。
    MAX_RSS_GB:      float = 12.0
    MEM_EXIT_CODE:   int   = 42
