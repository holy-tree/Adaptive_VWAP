+ results
	+ signals
		+ **方向信号预测结果（csv)**：boll（目录）, macd（目录）……
			+ **解释**：根据指标，生成两列`[日期，sell/buy]`
+ data
	+ latent, raw
+ models
	+ src
		+ model_with_clsdisc.py
		+ predictor_model.py
		+ **解释**：放置模型类
	+ pretrained 预训练好的模型
		+ vae_model
		+ predictor_model
+ scipts
	+ train_all_{lstm,rnn,Transformer,Vae}.sh
+ src
	+ core
		+ adaptive_vwap：核心计算逻辑，计算动态vwap，滑点，回报，回撤等，主要是调用函数
        + generate_signal 放置信号生成函数
		+ strategy：这里主要放每个阶段封装好的计算逻辑函数，给vwap调用
	+ utils
		+ data_utils：数据准备和处理
		+ model_utils
	+ vis：数据可视化
+ requirements.txt
+ README.md


