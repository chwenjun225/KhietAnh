from qutip import *
from qutip.qip.operations import hadamard_transform  # type: ignore
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# 1. Tạo trạng thái |0⟩
zero = basis(2, 0)
print("Trạng thái ban đầu:\n", zero)

# 2. Tạo cổng Hadamard (H)
H = hadamard_transform(1)

# 3. Áp dụng Hadamard lên |0⟩
psi = H * zero
print("\nSau khi áp dụng Hadamard:\n", psi)

# 4. Tính xác suất đo được |0⟩ và |1⟩
P0 = abs(psi.overlap(basis(2, 0)))**2
P1 = abs(psi.overlap(basis(2, 1)))**2
print(f"\nXác suất đo được |0⟩: {P0:.2f}")
print(f"Xác suất đo được |1⟩: {P1:.2f}")

# 5. Vẽ trạng thái trên Bloch Sphere
b = Bloch()
b.add_states(psi)
b.show()
