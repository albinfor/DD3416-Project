import numpy as np
import matplotlib.pyplot as plt 

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def ellipse(center, a, b= None, theta=0, N=300):
    if b == None: b=a;
    t = np.linspace(0, 2*np.pi, N)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    pts = R @ np.vstack((a*np.cos(t), b*np.sin(t)))
    return center[0] + pts[0], center[1] + pts[1]

def visualize_env(gx, gy, ax, ay):
    plt.plot(gx,gy, color='green')
    plt.plot(ax,ay, color='red')
    plt.fill(gx, gy, color='green', alpha=0.3, label='Goal set')
    plt.fill(ax, ay, color='red', alpha=0.3, label='Avoid set')
    ax = plt.gca()
    ax.set_xbound(-4,4)
    ax.set_ybound(-4,4)
    plt.grid()
    ax.set_aspect("equal")


def sdf_ellipse(X, Y, center, a, b = None, theta = 0):
    """
    Implicit ellipse SDF-like function.
    Negative inside, zero on boundary, positive outside.
    """
    if b == None: b=a;
    cx, cy = center

    # Translate
    Xc = X - cx
    Yc = Y - cy

    # Inverse rotation (rotate point by -theta)
    ct = np.cos(theta)
    st = np.sin(theta)

    Xr =  ct * Xc + st * Yc
    Yr = -st * Xc + ct * Yc

    # Axis-aligned ellipse implicit function
    return np.sqrt((Xr/a)**2 + (Yr/b)**2) - 1




def animate(Vs_reach, Vs_avoid, X, Y, phi_goal, phi_avoid, num_steps):
    def animation_func(k):    
        V_r = Vs_reach[k]    # from your reach solver
        V_a = Vs_avoid[k]    # from avoid solver above

        # Reach tube (controller winning) – blue
        plt.contour(X, Y, V_r, levels=[0], colors='b', linewidths=2)

        # Avoid tube (disturbance winning) – orange
        plt.contour(X, Y, V_a, levels=[0], colors='orange', linewidths=2)

        # Goal and obstacle geometry
        #plt.contour(X, Y, phi_goal,           levels=[0], colors='g',  linestyles='--')
        #plt.contour(X, Y, phi_avoid,     levels=[0], colors='r',  linestyles='--')
        
        plt.contourf(X, Y, phi_goal,           levels=[phi_goal.min(), 0], colors='b', alpha = 0.01)
        plt.contourf(X, Y, phi_avoid,     levels=[phi_avoid.min(), 0], colors='orange', alpha = 0.01)

        plt.gca().set_aspect('equal')
        plt.xlabel("x"); plt.ylabel("y")
        plt.title(f"Reach tube (blue) vs Avoid tube (orange), step {k}/{num_steps}")

    fig = plt.figure()
    return FuncAnimation(fig,animation_func,
                            frames=len(Vs_reach), interval=50, blit=False)
