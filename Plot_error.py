from Ridge_Reg import x_test, Y_test, Ridge_predicted, y_test
from matplotlib import cm
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
differance = Ridge_predicted - y_test
surf = ax.plot_trisurf(x_test,Y_test, differance, cmap=cm.coolwarm,linewidth=0, antialiased=False)
cbaxes = fig.add_axes([0.1, 0.3, 0.01, 0.4]) 
fig.colorbar(surf, shrink=0.5, aspect=5, cax = cbaxes )
ax.set_xlabel('Vdc1, volts')
ax.set_ylabel('Vdc2, volts')
ax.set_zlabel('Angle, degrees')
plt.show()