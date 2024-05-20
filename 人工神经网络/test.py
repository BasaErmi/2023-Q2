import module
import utils
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns

def print_metrics(precision, recall, macro_f1, class_names):
    table = PrettyTable()
    table.field_names = ["Class", "Precision", "Recall", "F1-Score"]

    for i, class_name in enumerate(class_names):
        table.add_row([class_name, f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{(2 * precision[i] * recall[i] / (precision[i] + recall[i])):.4f}"])

    print(table)
    print(f"Macro-F1 score: {macro_f1:.4f}")
    val_acc = utils.evaluate_accuracy(model, trainer, data.get_dataloader(False))
    print(f'Validation accuracy: {val_acc:.4f}')

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# 类名列表
class_names = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

"""加载数据集"""
data = module.EmotionDataModule(batch_size=32, resize=(224, 224))

"""加载模型"""
model = module.ResNeXt18(lr=0.005,num_classes=5)
checkpoint_path = 'checkpoints/best_model.pth'
model = utils.inference(model, checkpoint_path)

"""预测"""
trainer = module.Trainer(max_epochs=70, num_gpus=1)
precision, recall, macro_f1, cm = utils.evaluate_metrics(model, trainer, data.get_dataloader(False))

print_metrics(precision, recall, macro_f1, class_names)
