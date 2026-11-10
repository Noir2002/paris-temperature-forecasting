# 教程：巴黎月最高气温预测 —— 从一个过拟合的课程项目到一次严谨的评估

> 适用对象：准备 "LLM × 时间序列" 方向技术面试的我自己。
> 所有数字都来自实际运行输出：`results/summary.md`（由 `make all` 生成）和 `notebooks/walkthrough.ipynb` 的输出。
> 没有把握的地方会直接写"不确定"。

**目录**

1. [一句话结论](#1-一句话结论)
2. [知识清单](#2-知识清单)
3. [两条学习路线](#3-两条学习路线)（3.1 动手路线 · 3.2 速读路线）
4. [代码导读](#4-代码导读)
5. [结果解读和局限性](#5-结果解读和局限性)
6. [面试问答](#6-面试问答)
7. [2 分钟英文口述稿](#7-2-分钟英文口述稿)
8. [自测题](#8-自测题)

---

## 1. 一句话结论

**原项目在 12 个训练样本上，从 4,095 个滞后组合里按"调整 R² 最大 + 所有 p < 0.05"挑出的 9 滞后自回归模型是严重过拟合的：把同样的做法在 1980–2026 年每年重做一遍，预测 1–4 月的平均绝对误差（MAE）是 27.52 °C，而简单的 30 年气候平均只有 1.76 °C；即使给它 30 年数据（348 行），直接对原始温度做的滞后 AR 也赢不了气候平均（提前 2–4 个月显著更差），只有"气候平均 + 残差 AR"和 SARIMA 在提前 1 个月时略好（MAE 1.52 / 1.53 vs 1.62 °C）。**

另外：作业 PDF 给出的 2025 年 1–4 月真实值（7.5 / 8.6 / 14.6 / 20.0 °C）与 Météo-France 官方数据完全一致；原代码后来改用的 8 / 10 / 14 / 18 °C 与巴黎任何一个站都对不上，用它会得出相反的结论。

---

## 2. 知识清单

每个知识点按三段写：**是什么 → 最简单的公式或例子 → 在这个项目里为什么重要**。

### 2.1 趋势、季节性、平稳性（trend, seasonality, stationarity）

- **是什么**：时间序列 = 长期变化（趋势）+ 固定周期的重复（季节性）+ 剩下的随机波动。**平稳**（weakly stationary）是指均值、方差、自协方差不随时间变化。
- **例子**（练习 1 的输出）：蒙苏里站年平均最高气温 1951–1980 是 15.04 °C，1991–2020 是 16.48 °C，2016–2025 是 17.45 °C（趋势）；1991–2020 年 7 月比 1 月平均高 18.1 °C（季节性）。
- **为什么重要**：
  - 原始月温度**不平稳**：每个月的均值不同，而且整体在变暖。
  - 原始温度的 lag-12 自相关是 0.924，看起来"去年同月很有用"；但减去每月平均之后（得到**距平 anomaly**），lag-12 自相关只有 −0.045，lag-1 是 0.197。也就是说：**去年同月的信息几乎全部来自季节性本身，去年那个月的"偶然偏暖/偏冷"对今年没有预测力**。这一条解释了后面很多结果。
  - 变暖趋势让所有"用过去平均"的方法都偏冷（见 2.15）。

### 2.2 线性回归和最小二乘（linear regression, OLS）

- **是什么**：用一组特征的线性组合解释目标变量，系数选使残差平方和最小的那组。
- **公式**：$y = X\beta + \varepsilon$，$\hat\beta = \arg\min \sum (y_i - \hat y_i)^2 = (X^\top X)^{-1} X^\top y$。
- **例子**：原项目 Q12 用 2024 年最后 5 个月拟合出 `温度 = 58.02 − 4.58 × 月份编号`。外推到 2025 年 1 月（编号 12）得到 3.06 °C，而真实值 7.5 °C；继续外推到 4 月是 −10.68 °C。直线只在拟合区间内有意义，**不能外推一个周期性信号**。
- **为什么重要**：本项目所有模型（Q12 趋势线、各种 AR、距平 AR）本质都是 OLS。代码统一用 `statsmodels.OLS`，因为它直接给出 R²、调整 R²、AIC、BIC 和 p 值（scikit-learn 的 `LinearRegression` 不给 p 值，这也是原代码要手算 p 值的原因）。

### 2.3 R² 和调整 R²（R², adjusted R²）

- **是什么**：R² = 模型解释了多少比例的方差；调整 R² 对参数个数加了一点惩罚。
- **公式**：$R^2 = 1 - \frac{SSE}{SST}$，$R^2_{adj} = 1 - (1-R^2)\frac{n-1}{n-p-1}$（$n$ 样本数，$p$ 自变量个数）。
- **例子**：Q16 中 $n=12, p=9$，惩罚系数 $\frac{11}{2} = 5.5$。调整 R² = 0.99967，意味着未调整的 R² 约 0.99994（由公式反推）。惩罚看似很大，但只要残差足够小，R² 永远可以逼近 1。
- **为什么重要**：调整 R² 是**样本内**（in-sample）指标，它衡量"拟合得多好"，不衡量"预测得多好"。原项目把它当成选模标准，这是过拟合的直接原因。在 1980–2026 的重放中，每年选出的模型调整 R² 中位数是 1.000，样本外 MAE 却是 27.52 °C。

### 2.4 t 检验和 p 值（t-test, p-value）

- **是什么**：检验"某个系数其实为 0"（零假设 H0）。p 值 = 如果 H0 成立，看到至少这么极端的 t 统计量的概率。
- **公式**：$t = \hat\beta / \mathrm{SE}(\hat\beta)$，服从自由度 $n-p-1$ 的 t 分布。
- **例子**：Q14 的斜率 −4.58，p = 0.00113（只有 5 个点，自由度 3）。Q16 的 9 个斜率 p 值都 < 0.05，但截距的 p = 0.3316（原代码只检查斜率）。
- **为什么重要**：p 值**不是**"模型正确的概率"。它只有在"模型是事先定好的、只检验这一次"时才有意义。原项目先在 4,082 个模型里挑，再在同一批数据上算 p 值，这时 p 值已经失效（见 2.10）。

### 2.5 滞后项和自回归模型 AR(p)（lags, autoregression）

- **是什么**：用自己过去的值预测现在。
- **公式**：$y_t = c + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \varepsilon_t$。**子集 AR**（subset AR）只保留部分滞后，例如原项目的 {1, 2, 5, 7, 8, 9, 10, 11, 12}。
- **稳定性**：把系数写成特征多项式 $z^p - \phi_1 z^{p-1} - \dots - \phi_p = 0$，**最大根的模 < 1** 时冲击会衰减（稳定）；> 1 时递推预测会爆炸。代码里是 `ar.max_root_modulus`。
- **例子**（练习 9 的输出）：2025 年那个 9 滞后模型最大根 1.054（略微不稳定）；用 1988–1989 训练的模型最大根 3.558，递推到 1990 年 4 月预测出 −1,354 °C。用 30 年训练、2024-12 拟合的模型最大根 0.997（`results/04_rolling_fit_log.csv`；30 年窗口下各 AR 的最大根都不超过 1.000，接近 1 是因为季节周期本身就是一种“持续”的信号）。
- **为什么重要**：Q16 就是一个"在 12 个滞后里挑子集"的 AR。理解自由度和稳定性，就能解释为什么它会出离谱的预测。

### 2.6 递推多步预测 vs 直接预测（recursive vs direct multi-step）

- **递推**（recursive / iterated）：只训练一个"预测下一步"的模型，预测 2 月时把 1 月的**预测值**当输入，依此类推。
- **直接**（direct）：为每个步长 h 单独训练一个模型 $y_{t+h} = f_h(y_t, y_{t-1}, \dots)$，不把预测值喂回去。
- **误差为什么会累积**：递推时，第 1 步的误差进入第 2 步的输入；如果模型不稳定（根 > 1），误差还会被放大。
- **例子**（练习 9）：1990 年，递推的绝对误差是 23.19 → 112.55 → 381.99 → 1368.88 °C；如果每个月都用真实的上月值做一步预测，误差是 23.19, 25.08, 29.62, 16.34 °C（模型本来就很差，递推再把它放大了几十倍）。但在 2025 年，递推并没有更差（递推 0.45, 0.05, 2.64, 1.09；一步 0.45, 0.60, 2.28, 2.77）—— **累积是趋势，不是每次都发生**。
- **为什么重要**：本项目所有 AR 都用递推。滚动评估里 AR 的 MAE 从 h=1 的 1.69 升到 h=4 的 1.90 °C，而气候平均不依赖 h，一直是 1.62。直接预测法本项目没有实现（可作为后续工作）。

### 2.7 基线：朴素、季节性朴素、气候常年值（baselines）

| 名称 | 公式 | 本项目结果 |
|---|---|---|
| 朴素（naive） | $\hat y_{t+h} = y_t$ | 月温度有强季节性，基本没用，本项目未使用 |
| 季节性朴素（seasonal naive） | $\hat y_{t+h} = y_{t+h-12}$ | 滚动评估 MAE 2.12（最差的基线） |
| 两年同月平均 | $(y_{t+h-12} + y_{t+h-24})/2$ | 重放 MAE 1.97 |
| 气候常年值（climatology） | 过去 N 年同月平均 | 30 年：滚动 MAE 1.62，重放 1.76 |

- **为什么季节性朴素比气候平均差**：设 $y = \text{季节平均} + e$，$e$ 独立、方差 $\sigma^2$。季节性朴素的误差是 $e_t - e_{t-12}$，方差 $2\sigma^2$；气候平均（30 年平均几乎抵消了噪声）的误差约为 $e_t$，方差 $\sigma^2$。所以 RMSE 之比应接近 $\sqrt 2 = 1.414$。实测（练习 10）：2.660 / 2.025 = 1.314（略低于 √2，因为气候平均本身有变暖造成的偏差）。
- **为什么重要**：**任何复杂模型都必须先打败这些基线**。原项目只拿模型和真实值比，没有和基线比，所以看不出问题。

### 2.8 评估指标：MAE、RMSE（和偏差 bias）

- **公式**：$\mathrm{MAE} = \frac1n\sum|e_i|$，$\mathrm{RMSE} = \sqrt{\frac1n\sum e_i^2}$，$\mathrm{bias} = \frac1n\sum e_i$，$e_i = \hat y_i - y_i$。
- **区别**：RMSE ≥ MAE，对大误差更敏感。两者差距大说明有极端误差。
- **例子**：重放原方法，MAE 27.52、RMSE 115.25、中位绝对误差 4.80 °C —— 平均值被几次"爆炸"拉高，但即使看中位数也比气候平均（1.56）差三倍。
- **为什么重要**：只看一个指标会误导。bias 还揭示了所有方法都偏冷（2.15）。

### 2.9 过拟合和自由度（overfitting, degrees of freedom）

- **是什么**：模型记住了训练数据里的噪声，样本内误差很小、样本外误差很大。**残差自由度** $= n - p - 1$，是"拟合完之后还剩多少独立信息用来估计噪声"。
- **例子**：Q16 有 12 行、9 个斜率 + 1 个截距，自由度只剩 2。重放 1980–2026：样本内 RMSE 中位数 0.041 °C，样本外 MAE 27.52 °C。训练长度扫描（`results/fig5_window_sweep.png` 左图，样本内和样本外都是一步误差）：原规则在 2 年窗口的样本内 RMSE 0.227 vs 样本外 7.259；5 年 1.835 vs 2.332；10 年 1.951 vs 2.102；30 年 2.019 vs 2.116。差距在 5 年以后迅速缩小。
- **经验法则**：每个参数至少要 10–20 个样本。
- **为什么重要**：这是原项目的核心问题，而且是**样本量问题**，不是选模准则的问题：同一个规则在 30 年数据上几乎不过拟合。

### 2.10 多重比较和选择偏差（multiple comparisons, selection bias）

- **是什么**：检验很多次，总会有"显著"结果；从很多模型里挑样本内最好的，它的样本内表现会系统性地高估真实表现（winner's curse）。在同一批数据上"先选再检验"得到的 p 值不可信（post-selection inference）。
- **例子**（练习 5）：在 20 条**纯随机噪声**上跑原规则，17 条都能找到"所有系数 p < 0.05"的模型，其中 15 条调整 R² > 0.9，中位数 0.9994。噪声里没有任何可预测的东西，规则却"发现"了几乎完美的模型。
- **为什么重要**：这直接说明原项目的"调整 R² = 0.9997 且全部显著"**不构成任何证据**。

### 2.11 时间序列不能随机切分；滚动评估（no shuffling; rolling-origin evaluation）

- **为什么不能随机切分**：随机 K 折会让模型用"未来"的数据去预测"过去"。相邻月份相关、整体有趋势，所以随机切分会严重高估效果（**数据泄漏 data leakage**）。
- **滚动评估**（rolling-origin / time-series cross-validation）：对每个预测起点 $t$，只用 $\le t$ 的数据训练，预测 $t+1 \dots t+H$，然后把起点往后挪一个月。
- **本项目的设置**：起点 2000-12 到 2026-09（共 310 个），训练窗口为起点前 30 年，每年 12 月重新拟合（两次拟合之间冻结参数，只更新输入），预测 h = 1–4 个月。
- **怎么证明没有泄漏**：`tests/test_no_leakage.py` 把截止日之后的所有数据加 100 °C，断言截止日之前的预测**一位不差**；我还故意让代码多看一个月未来数据，这个测试确实失败了。
- **为什么重要**：原项目只有 4 个测试点，这里有 310 × 4 个，才能谈"显著"。

### 2.12 AIC / BIC（信息准则）

- **公式**：$\mathrm{AIC} = 2k - 2\ln L$，$\mathrm{BIC} = k\ln n - 2\ln L$（$k$ 参数个数，$L$ 似然）。OLS 下 $-2\ln L = n\ln(SSE/n) + $ 常数。$n \ge 8$ 时 $\ln n > 2$，BIC 惩罚更重、选的模型更小。
- **例子**：
  - 30 年窗口（348 行），26 次年度拟合的平均滞后个数：调整 R² 8.00、AIC 6.85、原规则 5.88、CV 6.04、BIC 4.73 —— 和理论一致。但样本外 MAE（h=1）几乎一样：1.686–1.692（CV 1.648）。
  - 12 行（练习 4）：调整 R²、AIC、BIC 全部选同一个 10 滞后模型（自由度 1）。样本太小时，残差可以被压到接近 0，$\ln(SSE/n) \to -\infty$，惩罚项挡不住。小样本应使用 AICc，或者干脆用更简单的模型。
- **为什么重要**：**选模准则救不了样本量**。样本够大时各准则差别很小，问题出在模型形式（见 2.14）。

### 2.13 季节模型：气候平均 + 残差 AR、SARIMA（补充）

- **气候平均 + 残差 AR**（`AnomalyAR`）：先减去训练窗口的每月平均得到距平，对距平拟合 AR(p)（p 由 AIC 在 0–12 中选），预测时再加回月平均。距平的 lag-1 自相关 0.197，所以短期有一点持续性可用，几个月后就退回气候平均。
- **SARIMA(1,0,0)(0,1,1)₁₂**：季节差分 + 季节 MA ≈ 一个"自适应的季节平均"，AR(1) ≈ 距平持续性。阶数**事先固定**，没有在测试集上调参。
- **结果**：两者在 h=1 显著优于 30 年气候平均（MAE 差 −0.098 和 −0.089 °C，95% 区间不含 0），h=2–4 与气候平均无显著差异。

### 2.14 为什么样本够大时原始滞后 AR 仍然输给气候平均（补充）

在原始温度上做 AR，季节性只能靠 lag-12、lag-11 等来表达，于是**去年那几个月的随机波动也被带进了预测**（就像部分的季节性朴素）。而距平的 lag-12 自相关是 −0.045，这部分信息只是噪声。气候平均对 30 年求平均，把噪声平均掉了。这是**模型形式（model specification）**的问题，换选模准则解决不了。

### 2.15 非平稳气候与偏差；block bootstrap（补充）

- **偏差**：滚动评估中所有方法 bias 为负（预测偏冷），30 年气候平均偏差 −0.84 °C，因为 30 年平均"落后"于变暖。训练长度扫描里，10 年窗口的气候平均（MAE 1.574）比 30 年（1.621）更准。30 年 AR 的残差按年代平均：1990 年代 −0.28，2000 年代 −0.22，2010 年代 0.09，2020 年代 0.49 °C（练习 7）—— 模型越来越低估。
- **Block bootstrap**：预测误差在相邻月份之间相关，普通 bootstrap 会低估不确定性。这里按 12 个月为一块重抽样，得到"方法 A 的 MAE − 气候平均的 MAE"的 95% 区间。

---

## 3. 两条学习路线

### 3.1 动手路线（约半天）

**准备**（10 分钟）：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/pip install jupyterlab && .venv/bin/jupyter lab notebooks/walkthrough.ipynb
```

每个练习都是：**先在纸上写预测 → 运行 notebook 对应单元 → 对照预期输出和检查点**。预期输出是我实际运行得到的。

---

**练习 1 数据核对**（20 分钟）

- 先预测：`data2.csv` 的 24 个值和蒙苏里站官方 `TX` 有几个相同？原始温度的 lag-12 自相关高吗？减去季节平均之后呢？
- 预期输出：
  ```
  922 months: 1950-01-01 -> 2026-10-01
  data2.csv == Montsouris TX: 24 / 24
  Jan-Apr 2025 (official): [ 7.5  8.6 14.6 20. ]
  annual mean TX 1951-1980: 15.04 °C
  annual mean TX 1991-2020: 16.48 °C
  annual mean TX 2016-2025: 17.45 °C
  seasonal amplitude (Jul - Jan normal): 18.1 °C | anomaly std: 1.84
  autocorrelation raw:       lag1 = 0.811  lag12 = 0.924
  autocorrelation anomalies: lag1 = 0.197  lag12 = -0.045
  ```
- 检查点：能说出 `TX` 字段的含义（日最高气温的月平均），以及为什么 0.924 的 lag-12 自相关"是季节性造成的，不是可利用的记忆"。

**练习 2 自己算季节性朴素基线**（15 分钟）

- 先预测：2025 年 1–4 月重复 2024 年同月，MAE 是多少？换成代码里的 8/10/14/18 呢？
- 预期输出：`forecast: [ 7.6 11.9 14.3 16.9]`，`MAE vs official: 1.7 | MAE vs code values: 0.93`
- 检查点：同一个预测，换一组"真实值"，MAE 从 1.70 变成 0.93 —— 4 个点的评估极其脆弱。

**练习 3 数一数原项目的穷举**（15 分钟）

- 先预测：12 个滞后有多少非空子集？12 行训练数据能拟合多少个？
- 预期输出：`subsets: 4095 | fitted: 4082 | training matrix: (12, 12)`，按子集大小：`{1: 12, 2: 66, ..., 9: 220, 10: 66}`
- 检查点：$2^{12}-1 = 4095$；11 个和 12 个滞后的 13 个子集（12 + 1）因为残差自由度 ≤ 0 被跳过。为什么训练矩阵只有 12 行？（滞后 12 需要前一年的数据，所以只有 2024 年的 12 个月能当目标。）

**练习 4 去掉 p 值筛选；改用 AIC / BIC**（25 分钟）

- 先预测：会选更多还是更少滞后？
- 预期输出：
  ```
  original  lags=(1, 2, 5, 7, 8, 9, 10, 11, 12) adjR2=0.99967 df_resid=2 forecast=[ 7.95  8.65 11.96 18.91] MAE=1.06
  adj_r2    lags=(1, 2, 5, 6, 7, 8, 9, 10, 11, 12) adjR2=0.99998 df_resid=1 forecast=[ 7.82  8.33 11.81 19.3 ] MAE=1.02
  aic       （同 adj_r2）
  bic       （同 adj_r2）
  ```
- 检查点：去掉 p 值、换 AIC、换 BIC，结果都是"把能用的自由度用到只剩 1"。而且这个更过拟合的模型在 2025 年 MAE 反而更低（1.02）—— 再次说明 4 个点什么也证明不了。

**练习 5 在纯噪声上跑原规则**（25 分钟）

- 先预测：20 条随机噪声里，有几条能"找到"全部显著的模型？
- 预期输出：
  ```
  noise series with >= 1 'all p < 0.05' model: 17 / 20
  ... of which adjusted R2 > 0.9: 15
  median best adjusted R2: 0.9994
  ```
- 检查点：能用"多重比较 + 选择偏差 + 自由度"三个词解释这个结果。这是面试里最有说服力的一个例子。

**练习 6 把最大滞后改成 6**（20 分钟）

- 先预测：训练行数会变吗？
- 预期输出：`subsets: 63 | training rows: 18`，`lags: (1, 4) | adjR2: 0.884 | df_resid: 15.0`，`forecast: [ 9.11 10.98 15.4  20.01] | MAE: 1.2`
- 检查点：最大滞后从 12 降到 6，训练行数从 12 增到 18（2023 年下半年也能当目标），选出的模型只有 2 个滞后、自由度 15，调整 R² 降到 0.884 —— **更"差"的样本内指标，往往是更诚实的模型**。

**练习 7 画残差图**（30 分钟）

- 先预测：12 行模型和 30 年模型，谁的样本内残差小？30 年模型的残差各月一样大吗？
- 预期输出：
  ```
  12-row model: max |resid| = 0.089 | in-sample RMSE = 0.048
  30-year model: (1, 4, 6, 8, 11, 12) | rows: 348 | in-sample RMSE: 1.99
  residual std by month: {1: 1.81, 2: 2.61, 3: 1.71, 4: 2.18, 5: 1.93, 6: 1.66, 7: 2.02, 8: 2.27, 9: 2.09, 10: 1.72, 11: 1.76, 12: 1.8}
  mean residual by decade: {1990: -0.28, 2000: -0.22, 2010: 0.09, 2020: 0.49}
  ```
- 检查点：12 行模型的残差小到 0.05 °C，这不是好事，是过拟合的信号（真实的月际随机波动大约 1.84 °C）。30 年模型的残差在 2 月最大（2.61）、6 月最小（1.66），说明误差**方差随季节变化**（异方差 heteroscedasticity）；按年代平均的残差从负变正，说明**变暖趋势没被模型捕捉**。

**练习 8 把原方法放到别的年份**（20 分钟）

- 先预测：如果课程在 1990 或 2020 年开，原 AR 的 MAE 是多少？
- 预期输出（每年 1–4 月 MAE，°C）：

  | year | ar_original | climatology_30y | seasonal_naive | two_year_mean |
  |---|---|---|---|---|
  | 1990 | 471.65 | 2.86 | 1.72 | 1.74 |
  | 2020 | 37.60 | 2.62 | 2.07 | 1.98 |
  | 2025 | 1.06 | 1.46 | 1.70 | 1.91 |

- 检查点：2025 年是少数"幸运年"之一。完整重放（47 年）里原 AR 只有 3 年比 30 年气候平均好（1989、2009、2025）。

**练习 9 递推误差累积**（25 分钟）

- 先预测：哪一年递推会出大问题？
- 预期输出：2025 年最大根 1.054，递推误差 `[0.45 0.05 2.64 1.09]`，一步误差 `[0.45 0.6 2.28 2.77]`；1990 年最大根 3.558，递推误差 `[23.19 112.55 381.99 1368.88]`，一步误差 `[23.19 25.08 29.62 16.34]`。
- 检查点：能解释"根的模 > 1 → 递推爆炸"；也能诚实地说 2025 年递推并没有更差。

**练习 10 最小的滚动评估**（25 分钟）

- 先预测：季节性朴素和 30 年气候平均谁更好？RMSE 之比？
- 预期输出：气候平均 MAE 1.619–1.624，季节性朴素 2.120–2.133（h=1–4）；`RMSE ratio (h=1): 1.314 | sqrt(2) = 1.414`
- 检查点：能用 2.7 节的 √2 论证解释，并能说出为什么气候平均的误差不随 h 变化。

**收尾**（15 分钟）：运行 `make test`（14 个测试，约 7 秒），读 `tests/test_no_leakage.py`，确认自己能讲清"腐蚀未来数据"的测试思路。

### 3.2 速读路线（30 分钟）

只读不跑。按这个顺序：

1. **第 1 节**一句话结论（2 分钟）。
2. **下面这个故事**（8 分钟）：
   - **起点**：2025 年 ISEP 数据科学课程作业，用 2023–2024 年巴黎 24 个月的最高气温预测 2025 年 1–4 月。我们穷举 4,095 个滞后组合，选出 9 个滞后的模型，调整 R² 0.9997，所有系数显著，2025 年 MAE 1.06 °C。
   - **疑点 1：真实值**。作业给的真实值和代码里的不一样，两者会给出相反的结论。我下载了 Météo-France 官方数据核实：`data2.csv` 正是蒙苏里站 `TX` 字段（24/24 一致），作业给的 2025 值是对的，代码里的 8/10/14/18 对不上任何站。
   - **疑点 2：过拟合**。12 行数据、10 个参数、自由度 2；在纯噪声上同样的规则 17/20 次能"找到"显著模型。把原方法在 1980–2026 年每年重做：MAE 27.52 °C，47 次拟合中 46 次不稳定（最大根 > 1），只有 3 年（1989、2009、2025）比气候平均好；2025 是幸运年。
   - **严格评估**：用 1950–2026 的长序列做滚动评估（310 个起点，h=1–4）。样本足够时过拟合基本消失，各选模准则结果几乎一样，但原始滞后 AR 仍输给气候平均；只有季节模型（距平 AR、SARIMA）在 h=1 显著更好，且优势只有约 0.1 °C。
   - **启示**：先建基线、先核实数据、按时间切分评估、警惕在小样本上搜索大量模型。
3. **看图**（5 分钟）：`results/fig2_expost_2025.png`（真实值问题）、`fig3_replay_original_protocol.png`（重放）、`fig5_window_sweep.png`（样本量）、`fig4_rolling_mae_by_horizon.png`（主结果）。
4. **知识清单**（10 分钟）：只读每节的"为什么重要"，重点 2.3、2.9、2.10、2.11、2.14。
5. **第 6 节**面试问答前 8 题（5 分钟）。

---

## 4. 代码导读

建议阅读顺序（核心 6 个文件约 660 行，30 分钟内可读完；`plots.py` 可以跳过）：

| 顺序 | 文件 | 行数 | 做什么 |
|---|---|---|---|
| 1 | `src/data.py` | 79 | 下载 Météo-France 文件，抽取蒙苏里站 `TX`，加载 `data2.csv`；作业和代码里的两组"真实值"常量 |
| 2 | `src/ar.py` | 118 | **核心**：滞后矩阵、穷举子集、四种选模准则、时间序列 CV、递推预测、稳定性检查 |
| 3 | `src/models.py` | 129 | 所有预测方法，统一接口 `fit(train)` / `forecast(history, steps)` |
| 4 | `src/original.py` | 49 | 用上面的模块复现原项目 Q12–Q17 |
| 5 | `src/evaluate.py` | 107 | 指标、滚动评估、原方法重放、block bootstrap |
| 6 | `src/run_all.py` | 179 | 一键跑完全部实验，写 `results/` |
| 7 | `src/plots.py` | 168 | 画图（颜色固定对应方法） |
| – | `src/verify_data.py` | 38 | 阶段 1 的数据核对（需要先下载原始文件） |
| – | `tests/` | – | 复现原数字、阶段 1 的数据事实、无未来数据泄漏 |

读代码时抓住三个设计：

1. **统一接口**：每个方法都有 `fit(train)` 和 `forecast(history, steps)`。`fit` 只看训练窗口；`forecast` 只看到起点为止的历史。评估代码不需要知道模型内部。
2. **选模和拟合分开**：`search_subsets` 一次拟合所有子集并记录调整 R²、AIC、BIC、最大 p 值，`select_lags` 按准则挑。所以"换准则"只是换一个字符串。
3. **泄漏防线在评估函数里**：`rolling_origin` 每个起点只切出 `series.loc[:origin]`，测试再从外部验证。

常用命令：

```bash
make all
```

```bash
make test
```

```bash
make data
```

---

## 5. 结果解读和局限性

### 5.1 主要结果

**2025 年 1–4 月事后评估**（只有 4 个点）

| 方法 | MAE（官方真实值） | RMSE（官方） | MAE（代码里的值） |
|---|---|---|---|
| 9 滞后 AR（原项目） | 1.06 | 1.45 | 1.09 |
| 1991–2020 常年值 | 1.40 | 1.95 | 1.05 |
| 季节性朴素 | 1.70 | 2.27 | 0.93 |
| 两年同月平均 | 1.91 | 2.34 | 0.81 |
| Q12 线性趋势外推 | 16.48 | 19.31 | 16.31 |

**原方法重放 1980–2026**（每年用前两年训练、预测 1–4 月，188 个点）

| 方法 | MAE | RMSE | 中位绝对误差 | 原 AR 更好的年数（共 47） |
|---|---|---|---|---|
| 原 AR | 27.52 | 115.25 | 4.80 | – |
| 30 年气候平均 | 1.76 | 2.18 | 1.56 | 3 |
| 两年同月平均 | 1.97 | 2.49 | 1.65 | 4 |
| 季节性朴素 | 2.30 | 2.95 | 1.95 | 7 |
| Q12 线性趋势 | 15.54 | 18.20 | 14.35 | 34 |

47 次拟合中有 46 次的最大根模 > 1（不稳定）。

**滚动评估 2001–2026，30 年窗口**（MAE，°C）

| 方法 | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| 气候平均 + 残差 AR | **1.521** | **1.583** | **1.595** | 1.620 |
| SARIMA(1,0,0)(0,1,1)₁₂ | 1.529 | 1.590 | 1.599 | **1.605** |
| 30 年气候平均 | 1.619 | 1.620 | 1.622 | 1.624 |
| AR，时间序列 CV | 1.648 | 1.816 | 1.829 | 1.848 |
| AR，BIC | 1.686 | 1.852 | 1.880 | 1.900 |
| AR，原规则 | 1.692 | 1.845 | 1.876 | 1.897 |
| AR，AIC | 1.692 | 1.851 | 1.881 | 1.900 |
| AR，调整 R²（无 p 值） | 1.692 | 1.855 | 1.880 | 1.903 |
| 季节性朴素 | 2.120 | 2.124 | 2.126 | 2.133 |

与 30 年气候平均的 MAE 差（95% block bootstrap 区间）：
- 原规则 AR：h=1 +0.073 [−0.014, 0.195]（不显著）；h=2 +0.225 [0.116, 0.342]；h=4 +0.273 [0.144, 0.388]（显著更差）。
- 距平 AR：h=1 −0.098 [−0.133, −0.019]（显著更好）；h=4 −0.004 [−0.030, 0.038]（无差异）。
- SARIMA：h=1 −0.089 [−0.123, −0.010]；h=4 −0.019 [−0.052, 0.038]。

**训练长度扫描**（同一测试期，h=1–4 合并的 MAE）

| 窗口 | 原规则 AR | AR (BIC) | 同窗口气候平均 | 距平 AR |
|---|---|---|---|---|
| 2 年 | 11.698 | 12.987 | 1.799 | 6.259 |
| 5 年 | 1.904 | 1.914 | 1.600 | 1.643 |
| 10 年 | 1.763 | 1.772 | 1.574 | 1.558 |
| 20 年 | 1.795 | 1.822 | 1.582 | 1.564 |
| 30 年 | 1.827 | 1.829 | 1.621 | 1.580 |

### 5.2 回答阶段 2 的两个问题

1. **样本足够多时，原选模方法会不会过拟合？** 基本不会。30 年窗口下原规则平均选 5.88 个滞后，样本内一步 RMSE 2.019，样本外一步 RMSE 2.116；不同准则（调整 R²、AIC、BIC、CV）的样本外 MAE 在每个 h 上只差 0.04–0.06 °C。训练长度扫描显示，一步误差的样本内外差距在 2 年窗口是 0.227 vs 7.259，5 年 1.835 vs 2.332，10 年以上约 0.08–0.15。过拟合是小样本造成的，不是规则本身在大样本下的问题。
2. **简单基线能不能打败它？** 能。30 年气候平均在 h=2–4 显著优于所有原始滞后 AR，h=1 无显著差异。原因是模型形式（2.14），不是选模准则。

### 5.3 局限性

- **只有一个站、一个变量**：结论只适用于巴黎蒙苏里站的月平均最高气温。
- **2025 年只有 4 个测试点**：任何方法间的差距都分不出来；这就是要做重放和滚动评估的原因。
- **滚动评估的误差不独立**：相邻起点、不同 h 的预测共享数据；block bootstrap 只是近似处理。
- **测试期有变暖趋势**：所有方法都偏冷；我没有测试"带趋势的气候平均"等可能更好的模型。
- **模型设定**：SARIMA 阶数事先固定，距平 AR 用 AIC 选阶；每年 12 月重新拟合、期间冻结参数。其他设定可能更好，我没有搜索（刻意避免在测试集上调参）。
- **CV 只针对一步误差**：`ar_cv` 用的是一步预测的验证误差，没有直接优化 4 步误差。
- **真实值来源**：官方数据质量码为 1（已验证），但不是 0（最终保护），理论上还可能被 Météo-France 修订；数据快照日期 2026-11-10。
- **代码里 8/10/14/18 的来源仍然不明**：我只能确认它们不是巴黎任何站的 `TX`，不知道它们从哪里来。
- **没有外部变量**：如北大西洋涛动（NAO）、海温等，可能对月尺度预测有帮助。

---

## 6. 面试问答

每题：**英文简答**（可以直接说）+ 中文解释（为什么这样答）。

**Q1. Walk me through this project.**
> It started as a 2025 course project: forecast Paris monthly maximum temperature for Jan–Apr 2025 from 24 months of data. We picked a 9-lag autoregression out of 4,095 lag subsets by adjusted R². In 2026 I re-did it properly: I verified the data against Météo-France, replayed the original method on 47 past years, and ran a rolling-origin backtest on monthly data since 1950. The original model was badly overfit; a 30-year climatology beats it, and only seasonal models beat climatology, by about 0.1 °C one month ahead.

中文：先说问题，再说"我发现了什么问题、怎么验证"，最后给数字。用第 7 节的口述稿展开。

**Q2. Did you write the code yourself?**
> Partly. The original course code was mostly AI-generated and, honestly, I didn't fully understand it then. For the 2026 rework I again used an AI coding assistant to write much of the code, but I defined the questions, checked the data against the official source, and verified every number by re-running it. I can explain each design choice — for example the leakage test, which corrupts all data after a cutoff and requires earlier forecasts to stay identical.

中文：诚实、具体。说清楚"AI 写了什么"和"我核实了什么"。**前提是你真的做了第 3.1 节的练习**，能当场解释任何一个函数。不要夸大，也不要过度自贬；重点是判断力和验证能力。

**Q3. What was wrong with the original model?**
> Twelve training rows and ten parameters, so two residual degrees of freedom. It was selected from 4,082 fitted candidates by an in-sample criterion, and the p-values were computed on the same data. When I replay the exact protocol on 1980–2026, its MAE is 27.5 °C versus 1.76 °C for a 30-year climatology.

中文：三个问题——自由度、选择偏差、样本内指标。用重放数字收尾。

**Q4. The adjusted R² was 0.9997. Why isn't that good news?**
> Adjusted R² measures in-sample fit. With 2 residual degrees of freedom almost any curve fits. On pure white noise the same procedure finds models with adjusted R² above 0.9 in 15 of 20 runs.

中文：用噪声实验（练习 5）一招制胜。

**Q5. But all coefficients had p < 0.05.**
> Those p-values assume the model was fixed before looking at the data. Here it was chosen as the best of thousands, so the tests are invalid — that's post-selection inference. On noise, 17 of 20 series produce an "all significant" model.

中文：多重比较 + 选择偏差。p 值不是"模型正确的概率"。

**Q6. How did you find and resolve the ground-truth inconsistency?**
> The assignment PDF gave 7.5/8.6/14.6/20.0 °C, but a later commit in our code changed them to 8/10/14/18 with no source, and the two versions flip the conclusion. I downloaded Météo-France's monthly climatological data, identified that our training file is exactly station Paris-Montsouris, field TX — 24 of 24 values match — and the official 2025 values equal the PDF's. The code's values match no Paris station.

中文：展示"核实数据来源"的习惯。这一条也解释了为什么简历结论要改。

**Q7. Why can't you shuffle time series for cross-validation?**
> Shuffling lets the model train on the future to predict the past, and autocorrelation and trends leak information. I used rolling-origin evaluation: at each monthly origin, fit only on data up to that month and forecast the next one to four months — 310 origins from 2000 to 2026.

**Q8. How do you know there is no look-ahead leakage?**
> There's a test: add 100 °C to every observation after a cutoff and assert that all forecasts made before the cutoff are bit-for-bit identical. I also checked the test has power — if I let the code see one extra month, it fails.

中文：这是"工程 taste"的好例子：不是口头保证，而是可执行的测试，并且测试了测试本身（mutation test）。

**Q9. Why is the seasonal naive forecast worse than climatology?**
> If temperature is a seasonal mean plus independent noise, seasonal naive carries last year's noise, so its error variance is about twice that of the climatology. Expected RMSE ratio √2 ≈ 1.41; I measured 1.31.

**Q10. With 30 years of data, why does the lag-based AR still lose to climatology?**
> Selection isn't the issue anymore — adjusted R², AIC, BIC and CV give almost identical results. It's the specification: modelling raw temperature, seasonality has to come through lags 11–12, which also imports last year's random anomalies. After removing the seasonal mean, the lag-12 autocorrelation is about zero (−0.045). Modelling anomalies instead fixes it.

**Q11. Recursive versus direct multi-step forecasting?**
> Recursive fits one one-step model and feeds predictions back in, so errors can compound — with an unstable fit (root modulus 3.6) a forecast reached −1,354 °C. Direct fits one model per horizon: no feedback, but more models and fewer effective samples. I used recursive throughout; a direct comparison is future work.

中文：要诚实说明没实现直接法；同时指出 2025 年递推并没有更差（累积是趋势不是必然）。

**Q12. AIC, BIC, adjusted R², or cross-validation — which would you use?**
> With 348 rows they disagree on the number of lags (BIC about 4.7, adjusted R² 8) but forecast almost equally well. With 12 rows none of them works. I'd prefer time-series cross-validation on the actual forecast horizon, plus a simple, physically motivated model class — and always a baseline.

**Q13. Are the differences between methods significant?**
> On the four 2025 points, no — even swapping the truth values flips the ranking. On the rolling backtest I used a 12-month block bootstrap: seasonal models beat climatology at one month ahead (about −0.09 to −0.10 °C, CI excluding zero) but not at two to four months; lag AR is significantly worse at two to four months.

**Q14. Why does everything have a negative bias?**
> Warming. A 30-year mean lags the current climate; its bias was −0.84 °C. A 10-year window was more accurate than 30 years in my sweep, which suggests adding a trend term or a shorter window.

**Q15. If you had 30 (or more) years of data, what would you do?**
> Exactly what the backtest does, but I'd go further: model anomalies relative to a trend-aware climatology (e.g., month effects plus a slow trend), use rolling-origin CV on the target horizon to pick the few hyperparameters, evaluate with block-bootstrap intervals, and add predictors with known monthly skill, such as large-scale circulation indices — and compare everything to climatology and seasonal naive.

中文：体现"先基线、再结构、再特征"的顺序；承认月尺度可预测性本来就有限（最好模型只比气候平均好约 0.1 °C）。

**Q16. Could LLMs or time-series foundation models like Chronos or TimesFM help here?**
> They're worth testing as zero-shot baselines — they're pretrained on many series and handle seasonality without hand-crafted lags. But with one monthly series, the signal beyond climatology is tiny, so I'd expect gains to be small and I'd evaluate them with the same rolling-origin protocol, watch for pretraining-data overlap with public weather series, and compare against climatology. LLMs could help more on the workflow side — for example checking metadata or generating code — than as the forecaster itself.

中文：这道题考"不盲目追新"。要点：①可以作为零样本（zero-shot）基线；②同一评估协议；③数据泄漏风险（预训练语料可能包含公开气象数据）；④对月度单站数据，可预测信号很少，预期收益有限；⑤LLM 更适合做流程辅助。只讨论，不实现。

**Q17. What are the limitations of your evaluation?**
> One station and one variable; only four points for 2025; overlapping forecasts, so errors are correlated; a warming test period; fixed SARIMA orders that I deliberately didn't tune on the test set; and the CV criterion only targets one-step errors.

**Q18. Why SARIMA(1,0,0)(0,1,1)₁₂? Did you tune it?**
> No, I fixed it in advance to avoid tuning on the test period. Seasonal differencing plus a seasonal MA behaves like an adaptive seasonal mean, and the AR(1) captures short anomaly persistence — the two things the data analysis suggested.

**Q19. What would you do differently if you redid the original project?**
> Verify the data source first, build seasonal naive and climatology baselines before any model, use far fewer parameters than data points, evaluate out of sample with a time-ordered split, and never pick a model by adjusted R² over thousands of candidates.

**Q20. What did you learn?**
> That a beautiful in-sample number is a warning, not a result, and that "where does this number come from?" is a question worth asking about every value — including the ground truth.

---

## 7. 2 分钟英文口述稿

> **Problem.** In a 2025 data-science course project, my team had to forecast Paris monthly maximum temperatures for January to April 2025, using only 24 months of data. We searched all 4,095 subsets of twelve monthly lags and picked a nine-lag regression with an adjusted R² of 0.9997. It looked great — but it had only two residual degrees of freedom, and our "true" 2025 values had been changed in the code without a source.
>
> **Approach.** In 2026 I redid it properly. First, data: I downloaded Météo-France's official monthly data and confirmed our file is exactly the Paris-Montsouris series — 24 out of 24 values — and that the official 2025 values were the assignment's, not the ones in our code. Second, I reproduced the original numbers exactly, with tests. Third, I evaluated out of sample: I replayed the original protocol on every year from 1980 to 2026, and I ran a rolling-origin backtest from 2001 to 2026 with a 30-year window, comparing seasonal naive, climatology, lag-based AR selected by adjusted R², AIC, BIC and cross-validation, and two seasonal models. A unit test guarantees no future data is used.
>
> **Results.** The original method was badly overfit: replayed over 47 years, its mean absolute error is 27.5 degrees, against 1.76 for a 30-year climatology; 2025 was one of only three lucky years. On pure noise, the same procedure finds a "significant" model 17 times out of 20. With 30 years of data the overfitting mostly disappears and the selection criterion barely matters — but the lag model still loses to climatology two to four months ahead. Only seasonal models, an anomaly AR and a SARIMA, beat climatology, and only one month ahead, by about 0.1 degree.
>
> **What I'd change.** Verify data first, start with baselines, model anomalies instead of raw temperature, and choose models out of sample. The main lesson: an in-sample R² of 0.9997 is a warning sign, not a result.

（约 320 词，正常语速约 2 分钟。）

---

## 8. 自测题

**1. 原项目 Q16 的训练矩阵为什么只有 12 行？**
<details><summary>答案</summary>最大滞后是 12，每一行需要前 12 个月的值，24 个月数据里只有 2024 年的 12 个月能当目标。</details>

**2. Q16 的残差自由度是多少？怎么算的？**
<details><summary>答案</summary>12 − 9 − 1 = 2（12 行、9 个斜率、1 个截距）。</details>

**3. 为什么 4,095 个组合里实际只拟合了 4,082 个？**
<details><summary>答案</summary>11 个滞后（12 种）和 12 个滞后（1 种）的组合会让残差自由度 ≤ 0，被跳过；12 + 1 = 13。</details>

**4. 写出调整 R² 的公式，并说明它为什么不能防止过拟合。**
<details><summary>答案</summary>R²_adj = 1 − (1 − R²)(n − 1)/(n − p − 1)。它仍是样本内指标；只要残差足够小就能接近 1，而且在很多候选里取最大会产生选择偏差。</details>

**5. 季节性朴素和 30 年气候平均的 RMSE 之比理论上约为多少？为什么？实测是多少？**
<details><summary>答案</summary>约 √2 ≈ 1.41：季节性朴素的误差是两个独立噪声之差，方差翻倍。实测 2.660 / 2.025 = 1.314。</details>

**6. 时间序列为什么不能随机 K 折？本项目怎么评估？**
<details><summary>答案</summary>会用未来预测过去，产生泄漏。本项目用滚动评估：310 个月度起点，只用起点之前 30 年数据训练，预测 1–4 个月，每年 12 月重新拟合。</details>

**7. 怎么用测试证明没有用到未来数据？**
<details><summary>答案</summary>把截止日之后的数据全部加 100 °C，断言截止日之前起点的预测完全不变；并且故意引入一个月泄漏，确认测试会失败。</details>

**8. AR 模型"最大根模 > 1"意味着什么？举本项目的例子。**
<details><summary>答案</summary>模型不稳定，冲击会放大，递推预测可能爆炸。1988–1989 训练的模型最大根 3.558，递推预测 1990 年 4 月为 −1,354 °C（真实 14.6 °C）。</details>

**9. 在 30 年窗口下，换用 AIC / BIC / CV 选滞后能不能让原始滞后 AR 打败气候平均？为什么？**
<details><summary>答案</summary>不能。h=1 时各准则 MAE 在 1.648–1.692，气候平均 1.619；h=2–4 都显著更差。问题在模型形式：原始温度上的滞后把去年的随机距平带进预测，而距平的 lag-12 自相关约为 0（−0.045）。</details>

**10. 2025 年 1–4 月的真实值是多少？来源是什么？为什么代码里的值会改变结论？**
<details><summary>答案</summary>7.5 / 8.6 / 14.6 / 20.0 °C，来源 Météo-France 开放数据，蒙苏里站（75114001）字段 TX，与作业 PDF 一致。用代码里的 8/10/14/18，季节性朴素（0.93）和两年平均（0.81）的 MAE 都比 AR（1.09）低；用官方值则 AR（1.06）最低。4 个点上排名完全取决于真实值，说明这种评估本身没有说服力。</details>
