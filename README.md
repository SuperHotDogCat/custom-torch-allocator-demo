# custom-torch-allocator-demo
PytorchのCUDAメモリ割当を変更するにはどうすればよいのかのデモ<br>
# 実行環境
Ubuntu: 22.04<br>
nvcc: 12.3<br>
CUDA Version: 12.4<br>
Driver Version: 550.127.05<br>
GPU: RTX4060 laptop<br>

CUDA Unified Memoryが有効なバージョンとGPUにしておくのが良いです<br>
それ以外の環境構築は,,, みんなに任せた！<br>
# 遊び方
以下のshell scriptを実行する<br>
```
chmod +x custom_allocator_experiment.sh
./custom_allocator_experiment.sh
```
memory allocatorの違いを眺める
```
g++ -shared -fPIC -O3 -I/usr/local/cuda/include -c allocator.cpp -o allocator.o
g++ -shared allocator.o -L/usr/local/cuda/lib64 -lcudart -o allocator.so
Normal cudaMalloc Test
First CPU Memory: 5773.38 MiB -> 5773.38 MiB / 15601.39 MiB
First GPU Memory: 216.00 MiB -> 216.00 MiB / 8188.00 MiB
Epoch [100/100], Loss: 0.0000
After CPU Memory: 6274.54 MiB -> 6297.69 MiB / 15601.39 MiB
After GPU Memory: 823.00 MiB -> 823.00 MiB / 8188.00 MiB
Elapsed time: 3.42899227142334

rm -f allocator.o allocator.so
g++ -shared -fPIC -O3 -DMANAGED -I/usr/local/cuda/include -c allocator.cpp -o allocator.o
g++ -shared allocator.o -L/usr/local/cuda/lib64 -lcudart -o allocator.so
cudaManagedMalloc Test
First CPU Memory: 5852.64 MiB -> 5852.64 MiB / 15601.39 MiB
First GPU Memory: 216.00 MiB -> 216.00 MiB / 8188.00 MiB
Epoch [100/100], Loss: 0.0000
After CPU Memory: 6206.38 MiB -> 6222.11 MiB / 15601.39 MiB
After GPU Memory: 383.00 MiB -> 475.00 MiB / 8188.00 MiB
Elapsed time: 7.411806583404541

rm -f allocator.o allocator.so
g++ -shared -fPIC -O3 -DCPU -I/usr/local/cuda/include -c allocator.cpp -o allocator.o
g++ -shared allocator.o -L/usr/local/cuda/lib64 -lcudart -o allocator.so
cudaMallocHost Test
First CPU Memory: 5886.97 MiB -> 5886.97 MiB / 15601.39 MiB
First GPU Memory: 222.00 MiB -> 222.00 MiB / 8188.00 MiB
Epoch [100/100], Loss: 0.0000
After CPU Memory: 6351.79 MiB -> 6338.65 MiB / 15601.39 MiB
After GPU Memory: 349.00 MiB -> 349.00 MiB / 8188.00 MiB
Elapsed time: 32.97867798805237
```
