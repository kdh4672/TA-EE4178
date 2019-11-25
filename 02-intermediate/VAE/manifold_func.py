import numpy as np


def plt_manifold(trained_model_instance, mean_range=3, n=20, figsize=(8, 10)): #
    x_axis = np.linspace(-mean_range, mean_range, n)
    y_axis = np.linspace(-mean_range, mean_range, n)
    canvas = np.empty((28*n, 28*n))

    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mean = np.array([[xi, yi]] * 1)
            z_mean = torch.tensor(z_mean, device=device).float()
            x_reconst = model.decode(z_mean)
            canvas[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_reconst[0].reshape(28, 28)

    plt.figure(figsize=figsize)
    xi, yi = np.meshgrid(x_axis, y_axis)
    plt.imshow(canvas, origin="upper")
    plt.savefig(manifold_image)
    plt.show()

    return

""" 적용예시

with torch.no_grad():
    plt_manifold(model)
"""
