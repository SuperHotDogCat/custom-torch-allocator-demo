from time import time
import psutil
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
# memory allocator 変更
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(f'allocator.so', 'custom_malloc', 'custom_free')
torch.cuda.memory.change_current_allocator(new_alloc)

# GPUメモリ使用量を取得する関数
def get_gpu_memory_usage():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        used, total = map(int, result.strip().split("\n")[0].split(","))
        return used, total
    except Exception as e:
        print(f"GPU memory retrieval failed: {e}")
        return None, None

# CPUメモリ使用量を取得する関数（MiB表記）
def get_cpu_memory_usage():
    mem = psutil.virtual_memory()
    return mem.used / (1024 ** 2), mem.total / (1024 ** 2)

# MLPモデル定義
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ハイパーパラメータ
input_size = 10
hidden_size = 1000000
output_size = 5
batch_size = 32
epochs = 100
learning_rate = 0.01
cpu_mem, cpu_mem_total = get_cpu_memory_usage()
gpu_mem, gpu_mem_total = get_gpu_memory_usage()
print(f"First CPU Memory: {cpu_mem:.2f} MiB -> {cpu_mem:.2f} MiB / {cpu_mem_total:.2f} MiB")
print(f"First GPU Memory: {gpu_mem:.2f} MiB -> {gpu_mem:.2f} MiB / {gpu_mem_total:.2f} MiB")

# ダミーデータ
x = torch.randn(batch_size, input_size).to("cuda:0")
y = torch.randint(0, output_size, (batch_size,)).to("cuda:0")

# モデル、損失関数、最適化器の初期化
model = SimpleMLP(input_size, hidden_size, output_size).to("cuda:0")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
start = time()
# 学習ループ
for epoch in range(epochs):

    cpu_mem_before, cpu_mem_total = get_cpu_memory_usage()
    gpu_mem_before, gpu_mem_total = get_gpu_memory_usage()
    # 順伝播
    outputs = model(x)
    loss = criterion(outputs, y)

    # 逆伝播
    optimizer.zero_grad()  # 勾配を初期化
    loss.backward()        # 逆伝播
    optimizer.step()       # 重みを更新

    cpu_mem_after, _ = get_cpu_memory_usage()
    gpu_mem_after, _ = get_gpu_memory_usage()

# 結果を出力
print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print(f"After CPU Memory: {cpu_mem_before:.2f} MiB -> {cpu_mem_after:.2f} MiB / {cpu_mem_total:.2f} MiB")
print(f"After GPU Memory: {gpu_mem_before:.2f} MiB -> {gpu_mem_after:.2f} MiB / {gpu_mem_total:.2f} MiB")
end = time()
print(f"Elapsed time: {end - start}")
