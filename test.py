import numpy as np
label = np.ones(5) * -1


normal_label = np.copy(label)
normal_label[-1] = 0
normal_label = np.repeat(np.expand_dims(normal_label, 0), 2, 0)
print(label)
print(normal_label)