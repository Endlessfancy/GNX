补充一点关于 NPU 模型复用的逻辑：

我考虑到频繁加载 NPU 模型开销太大，所以我们采用 Global Max Padding 策略。

Model Export: 在初始化阶段，利用 data_loader.get_max_size() 获取整个数据集的全局最大 max_nodes 和 max_edges。只导出 一个 使用这些最大维度的 NPU 静态模型。

Model Loading: get_model_path 应该总是去查找这个全局最大尺寸的模型，而不是查找当前 Batch 实际尺寸的模型。

Inference: 运行时，所有 Batch 的数据都必须先 Pad 到这个全局最大尺寸，再送入 NPU。

请确保代码逻辑（特别是 get_model_path 的调用处和 export 处）是使用 self.max_nodes (全局最大值)，而不是循环中的 batch_num_nodes。