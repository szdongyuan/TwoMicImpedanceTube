import matplotlib.pyplot as plt
import numpy as np
import mplcursors

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots()
(line1,) = ax.plot(x, y1, label="sin(x)")
(line2,) = ax.plot(x, y2, label="cos(x)")
ax.legend()

# 只给 sin(x) 曲线添加 cursor
cursor = mplcursors.cursor(line1, hover=True)


# 自定义显示格式
@cursor.connect("add")
def on_add(sel):
    x_val, y_val = sel.target
    sel.annotation.set_text(f"x={x_val:.2f}\ny={y_val:.2f}")


plt.show()
