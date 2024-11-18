make clean
make
echo "Normal cudaMalloc Test"
python simple_training_process.py
echo ""
make clean
make CXXFLAGS="-fPIC -O3 -DMANAGED"
echo "cudaManagedMalloc Test"
python simple_training_process.py
echo ""
make clean
make CXXFLAGS="-fPIC -O3 -DCPU"
echo "cudaMallocHost Test"
python simple_training_process.py
echo ""
