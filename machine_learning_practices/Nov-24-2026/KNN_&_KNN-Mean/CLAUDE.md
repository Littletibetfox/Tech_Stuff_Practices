# KNN 从零实现 · 共享笔记

> 这份文件是终端里的 Claude Code 和 claude.ai 项目对话之间的**共享记忆**。
> 两边都会读它；进度或约定有变化时，请直接更新本文件（尤其是“当前进度”和“学习记录”）。

## 学习方式
- 全程用**简体中文**交流，但**技术术语和关键动词必须用英文**（如 constructor、instance、method、call、return、import），不要翻译成中文。
- 辅导模式：每一步先讲原理和要做什么，让 LT 自己写代码，Claude 负责检查、指出问题、给提示。
- 除非 LT 明确要求，**不要直接给出完整答案**。
- LT 有 Java 背景，讲 Python 设计习惯时可以和 Java 对比。

## 数据：Data/knn.csv
- 600 行，表头 `feature_1,feature_2,label`（表头是后加的）
- feature_1 范围约 0.99~7.37，feature_2 约 0.03~2.70（尺度不同，后面要讨论标准化）
- label 为 0/1/2，每类约 200 个，无缺失值
- `Data/knn_original_noheader.csv` 是加表头前的原始备份

## 路线
1. 加载数据，得到 X (600, 2)、y (600,)，y 转 int
2. 打乱并划分训练集 / 测试集
3. 欧氏距离函数
4. 找最近的 k 个训练样本
5. 多数投票预测
6. 测试集准确率
7. 尝试不同 k 选最优，讨论特征标准化
8. （可选）画决策边界

## 代码结构
- `data_loader.py`：`CSVDataLoader` 类，`load()` 返回 `X, y`
- `knn.py`：`KNN` 类，`fit` / `predict`，只接收 X, y，不关心数据来源（依赖注入）
- `main.py`：串起来并评估

## 当前进度
- [x] 检查数据、加表头
- [ ] 第 1 步：写 data_loader.py 加载数据 ← 正在做

## 学习记录
- 已讲：Python 模块天然是单例（import 只加载一次，缓存在 sys.modules）；用独立的 loader 类 + 依赖注入隔离数据读取，通常比手写 Singleton 更推荐。
