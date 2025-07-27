# multiVAE
Multi VAE Toy project with MNIST dataset

Test and observe whether task vector editing and also show effect on VAE generation model.

## Experiment Flow

Define baseline model $VAE_{pre}$ as a VAE model that is trained with MNIST.

Define overfitted model $VAE_{n}$ as overfit model that is overfitted to digit n

Then, define task vectors for each digits as $\theta_{n}$ and weight of baseline model $\theta_{pre}$.

Here, let us observe the added model $\theta_{added} = \theta_{pre} + c_1 \theta_{i} + c_2 \theta_{j}$, can generate both digits i and j properly. 