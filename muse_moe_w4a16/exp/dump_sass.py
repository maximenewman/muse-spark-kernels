import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import submission as S
k = S._gemm(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
open(sys.argv[4], "wb").write(k.cubin.raw)
