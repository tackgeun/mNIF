nvcc = /usr/local/cuda/bin/nvcc
cudalib =  /usr/local/cuda/lib64
tensorflow = /home/tackgeun/anaconda3/envs/asym-diff/lib/python3.8/site-packages/tensorflow/include/

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++17 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX17_ABI=0


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX17_ABI=0 -std=c++17 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++17 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX17_ABI=0


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX17_ABI=0 -std=c++17 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
