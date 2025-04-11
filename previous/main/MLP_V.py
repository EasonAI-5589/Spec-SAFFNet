import os
import sys

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取父目录
parent_dir = os.path.dirname(current_dir)

# 切换到父目录
os.chdir(parent_dir)

# 将父目录添加到sys.path
sys.path.append(parent_dir)


from functions.data_preparation import *
from functions.draw import *
from functions.save_model import *
from models.Multilayer_Perceptron import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 说明：这是一段调试代码，但是可以用于直接检测一个元素





data_expert_path = './00数据集/dataset1.xlsx'

# 储存每个元素的data与label，并且储存用于训练的数据集
dataset = []
dataloader = []

# 存储每个元素对应的RMSE和R2
rmse_list = []
r2_list = []

# 读取元素，数据，与标签
data,label = data_reader(data_expert_path)
titles = title_reader(data_expert_path)


i = 3

data[i] = preprocessing(data[i])
data_train, data_test, labels_train, labels_test = train_test_split(data[i], label[i], test_size=0.1)

dataset_dict = {'data_train': data_train, 'data_test': data_test, 'labels_train': labels_train, 'labels_test': labels_test}

train_dataset,train_dataloader = data_loader(data_train,labels_train)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_train_tensor = torch.tensor(data_train).float().to(device)
labels_train_tensor = torch.tensor(labels_train).float().to(device)
data_test_tensor = torch.tensor(data_test).float().to(device)
labels_test_tensor = torch.tensor(labels_test).float().to(device)


# for i in range(0,len(data)):
#     data[i] = preprocessing(data[i])
#     x,y = data_loader(data[i],label[i])
#     dataset.append(x)
#     dataloader.append(y)

# input_features_dim = data[0].shape[1]
# 测试data[1]
input_features_dim = data[i].shape[1] # data的列数，代表了输入特征的维度
hidden_layer1_dim = 20
hidden_layer2_dim = 10
output_dim = 1


# 查看train_dataloader数据和标签的输出
# for inputs, targets in train_dataloader:
#     print(inputs.shape)
#     print(targets.shape)
#     break

# torch.Size([1, 6])
# torch.Size([1])


# 设置超参数

learning_rate = 0.0005
batch_size = 57 # 可以不考虑batch_size
num_epochs = 300

# 初始化模型和损失函数
model = Multilayer_Perceptron(input_features_dim, hidden_layer1_dim, hidden_layer2_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
consecutive_r2_count = 0

for epoch in range(num_epochs):
    optimizer.zero_grad()
    for inputs, targets in train_dataloader:
        targets = targets.reshape(-1,1) #这里本来shape是[1]， 但是为了符合神经网络[1,1]，我们将它reshape为[1,1],只改变了维度，没有改变数值
        outputs, intermediate_features = model(inputs)
        loss = criterion(outputs, targets) 
        loss.backward(retain_graph=True)
        optimizer.step()

        # 计算训练集上的预测精度差
        train_outputs,_ = model(data_train_tensor)
        train_outputs = train_outputs.reshape(-1) #同理，需要将输出的维度从[batchsize,outsize]reshape成为[outsize]
        train_outputs_np = train_outputs.detach().numpy().astype(np.float32)
        train_rmse = mean_squared_error(labels_train, train_outputs_np, squared=False)
        train_r2 = r2_score(labels_train, train_outputs_np)
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Train RMSE: {train_rmse}, Train R^2: {train_r2}")

    # 提取最后一次训练的隐藏层特征
    last_hidden_features = intermediate_features.detach().numpy().astype(np.float64)
    
    # 判断是否满足停止训练的条件
    if train_r2 > 0.99:
        consecutive_r2_count += 1
        if consecutive_r2_count >= 20:
            print("Training stopped: R^2 exceeded 0.99 for 20 consecutive epochs.")
            break
    # 重置计数器
    else:
        consecutive_r2_count = 0

# 测试模型

test_outputs,_ = model(data_test_tensor)
test_outputs = test_outputs.reshape(-1)
test_outputs_np = test_outputs.detach().numpy().astype(np.float32)
test_rmse = mean_squared_error(labels_test, test_outputs_np, squared=False)
test_r2 = r2_score(labels_test, test_outputs_np)

# print("For " + str(titles[i]) + ":" + f"Test RMSE: {test_rmse}, Test R^2: {test_r2}")
# print("---------------------------------------------------")


# 元素名称
title = str(titles[i])+": label vs prediction"

# 定义保存pickle文件和图片的路径
save_path = './results/MLPpklV/' # pickle文件
os.makedirs(save_path, exist_ok=True)
save_directory = './results/MLPV' # 图片
os.makedirs(save_directory, exist_ok=True)


# 加载已有字典，比较是否替换模型
model_info_path = f'./results/MLPpkl/model_info_' + str(titles[i]) + '.pkl'  # 请替换为实际的路径
with open(model_info_path, 'rb') as f:
    model_info = pickle.load(f)
    if model_info['test_r2'] < test_r2:

        # 画图并且保存
        draw_scatter(labels_test,test_outputs_np,test_rmse,test_r2,save_directory,"Sample","Prediction",title)

        save_model_info(str(titles[i]), save_path, model, train_rmse, train_r2, test_rmse, test_r2,dataset_dict)
        
        print("\n")
        print("For " + str(titles[i]) + ":")
        print("---------------------------------------------------")
        print("current test r2: " + str(test_r2) + " current test rmse: " + str(test_rmse))
        print("best test r2: " + str(test_r2) + " best test rmse: " + str(test_rmse))
        print("better model is saved")
        print("---------------------------------------------------")
        print("\n")

    else:
        print("\n")
        print("For " + str(titles[i]) + ":")
        print("---------------------------------------------------")
        print("current test r2: " + str(test_r2) + " current test rmse: " + str(test_rmse))
        print("best test r2: " + str(model_info['test_r2']) + " best test rmse: " + str(model_info['test_rmse']))
        print("current model is not saved")
        print("---------------------------------------------------")
        print("\n")

draw_scatter(labels_test,test_outputs_np,test_rmse,test_r2,save_directory,"Sample","Prediction",title)



import pickle
import torch
from models.Multilayer_Perceptron import Multilayer_Perceptron  # 导入你的模型


path = "/Users/guoyichen/Research_On_Mac/XRF光谱含量预测/results/MLPpkl/model_info_V.pkl"

with open(path, 'rb') as f:
    model_info = pickle.load(f)
    best_model_state_dict = model_info['model_state_dict']


best_model = Multilayer_Perceptron(input_features_dim, hidden_layer1_dim, hidden_layer2_dim, output_dim)

# 将加载的模型参数加载到模型中
best_model.load_state_dict(best_model_state_dict)

test_outputs,_ = best_model(data_test_tensor)
test_outputs = test_outputs.reshape(-1)
test_outputs_np = test_outputs.detach().numpy().astype(np.float32)
test_rmse = mean_squared_error(labels_test, test_outputs_np, squared=False)
test_r2 = r2_score(labels_test, test_outputs_np)

print("\n")
print("---------------------------------------------------")
print("best model:")
print("For " + str(titles[i]) + ":" + f"Test RMSE: {test_rmse}, Test R^2: {test_r2}")
print("---------------------------------------------------")