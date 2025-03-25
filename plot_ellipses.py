

from dataset.generate_ellipses import get_disk_dist_ellipses_dataset

import matplotlib.pyplot as plt 


dataset = get_disk_dist_ellipses_dataset(im_size=256, max_n_ellipse=70)


print(dataset)
print("Length: ", len(dataset))

fig, axes = plt.subplots(2,3, figsize=(12,6))

for idx, ax in enumerate(axes.ravel()):

    x = dataset[idx]
    
    ax.imshow(x[0].cpu().numpy(), cmap="gray")


plt.show()