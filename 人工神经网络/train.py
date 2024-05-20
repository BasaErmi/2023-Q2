import module
import utils

"""超参数"""
batch_size = 32
resize = (224, 224)
max_epochs = 100
num_gpus = 1
lr = 0.005
num_classes = 5
checkpoint_path = 'RessNeXt.pth'

"""加载数据集"""
data = module.EmotionDataModule(batch_size=batch_size, resize=resize)

"""加载模型"""
model = module.ResNeXt18(lr=lr, num_classes=num_classes)

"""开始训练"""
trainer = module.Trainer(max_epochs=max_epochs, num_gpus=num_gpus)
trainer.fit(model, data)