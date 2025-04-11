import os
import sys


# 切换到父目录
current_dir = os.path.dirname(os.path.abspath(__file__))    # 获取当前脚本的目录
parent_dir = os.path.dirname(current_dir)                   # 获取父目录
os.chdir(parent_dir)                                        # 切换到父目录
sys.path.append(parent_dir)                                 # 将父目录添加到sys.path



from functions.data_preparation import *
from functions.draw import *
from functions.save_model import *
from models.one_dim_CNN import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 说明：目前只是基于专家数据使用MLP模型进行训练
# 使用神经网络结构是：输入层-隐藏层1（20）-隐藏层2（10）-输出层（1）进行回归预测
# 五种元素的含量预测保存在results/MLP 文件中



# 读取所有元素的信息

data_expert_path = './00数据集/dataset2.xlsx' # 全光谱 Pb

# 数据说明

data,label = one_dim_data_reader(data_expert_path)

i = 0

data[i] = data[i][0:, :810]
data[i] = preprocessing(data[i])
data_train, data_test, labels_train, labels_test = train_test_split(data[i], label[i], test_size=0.1)
dataset_dict = {'data_train': data_train, 'data_test': data_test, 'labels_train': labels_train, 'labels_test': labels_test}
train_dataset,train_dataloader = data_loader(data_train,labels_train)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 在这个代码中，unsqueeze(1)在数据的第二个维度上增加了一个新的维度，使得数据的形状变为(sample_size, 1, feature_size)，
# 这是我们期望的形状。同时，标签通常不需要增加新的维度，所以我移除了对标签的unsqueeze和expand操作。
batch_size = 1  # 你的batch size
data_train_tensor = torch.tensor(data_train).float().unsqueeze(1).to(device)  # Change unsqueeze(0) to unsqueeze(1)
labels_train_tensor = torch.tensor(labels_train).float().to(device)  # Labels usually don't need to add a new dimension
data_test_tensor = torch.tensor(data_test).float().unsqueeze(1).to(device)  # Change unsqueeze(0) to unsqueeze(1)
labels_test_tensor = torch.tensor(labels_test).float().to(device)  # Labels usually don't need to add a new dimension


# 神经网络输入参数
input_size = data[i].shape[1]
hidden_size_1 =  1
hidden_size_2 = 30
conv_kernel_size =  3
pool_kernel_size = 3

# 设置训练参数
batch_size = 57
learning_rate = 0.002
num_epochs = 200

# 初始化模型和损失函数
model = one_dim_CNN(input_size, hidden_size_1, hidden_size_2, conv_kernel_size, pool_kernel_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    for inputs, targets in train_dataloader:
        targets = targets.reshape(-1,1) #这里本来shape是[1]， 但是为了符合神经网络[1,1]，我们将它reshape为[1,1],只改变了维度，没有改变数值
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad() 
        loss.backward(retain_graph=True)
        optimizer.step()

        # 计算训练集上的预测精度差
        train_outputs = model(data_train_tensor)
        train_outputs = train_outputs.reshape(-1) #同理，需要将输出的维度从[batchsize,outsize]reshape成为[outsize]
        train_outputs_np = train_outputs.detach().numpy().astype(np.float32)
        train_rmse = mean_squared_error(labels_train, train_outputs_np)
        train_r2 = r2_score(labels_train, train_outputs_np)
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Train RMSE: {train_rmse}, Train R^2: {train_r2}")


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

test_outputs = model(data_test_tensor)
test_outputs = test_outputs.reshape(-1)
test_outputs_np = test_outputs.detach().numpy().astype(np.float32)
test_rmse = mean_squared_error(labels_test, test_outputs_np)
test_r2 = r2_score(labels_test, test_outputs_np)

print("For " + "Pb" + ":" + f"Test RMSE: {test_rmse}, Test R^2: {test_r2}")
print("---------------------------------------------------")

save_directory = "./results/1D_CNN"


print(test_outputs_np)
print(labels_test)
draw_scatter(test_outputs_np, labels_test, test_rmse, test_r2, save_directory ,"Pb", "Prediction", "Prediction vs Real")
