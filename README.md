# iclr2023casa_tabular

The env is a very simple rectangular world, which aims to walk from left-top to the right-down. 

There is no functional approximation, but only Q-table, A-table and $\pi$-table, if needed. 

---

- /code/test.jpg shows the performance of a fully independent $\pi$, CASA, and MPO.
- /code/test_mix.jpg shows the performance of $\pi = softmax(\alpha \cdot \log \pi_\phi + (1.0 - \alpha) \cdot Q_\theta / \tau)$ with different $\alpha$.
