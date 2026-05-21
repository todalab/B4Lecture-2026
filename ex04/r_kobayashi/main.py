"""全データ向け実行用."""

import forward
import viterbi

print("forward\n")
for i in range(4):
    print(f"data:{i + 1}")
    forward.main(i + 1)
    print("\n")

print("vitervi\n")
for i in range(4):
    print(f"data:{i + 1}")
    viterbi.main(i + 1)
    print("\n")
