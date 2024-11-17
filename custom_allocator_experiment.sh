make clean
make
echo "Normal cudaMalloc Test"
rye run python simple_forward.py
echo ""
make clean
make CXXFLAGS="-fPIC -O3 -DMANAGED"
echo "cudaManagedMalloc Test"
rye run python simple_forward.py
echo ""
make clean
make CXXFLAGS="-fPIC -O3 -DCPU"
echo "cudaMallocHost Test"
rye run python simple_forward.py
echo ""
