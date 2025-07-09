This procedure computes a score that measures how parallel two specific hyperplanes are in a multi-dimensional space. The final value is derived from the absolute cosine similarity between the normal vectors of these two hyperplanes.

***

### 1. Defining the Hyperplanes 📐

The method compares the orientation of two distinct hyperplanes in an $N$-dimensional space ($\mathbb{R}^N$).

* **The Centroid Hyperplane**: The first hyperplane is dynamically defined by the data. The input points are grouped by their class labels, and a **centroid** (or mean vector) $\mathbf{c}_k$ is calculated for each of the $N$ classes. These $N$ centroid points $\{\mathbf{c}_1, \dots, \mathbf{c}_N\}$ uniquely define an $(N-1)$-dimensional hyperplane in the space. For this hyperplane to be unique, every class must have at least one sample point.

* **The Reference Hyperplane**: The second hyperplane is a fixed reference, defined by the simple equation $\sum_{i=1}^{N} x_i = 0$.

***

### 2. Finding the Normal Vectors 🧭

To compare the orientation of the hyperplanes, we must first find their respective normal vectors—vectors that are perfectly perpendicular to the surface of each hyperplane.

* **Centroid Normal via SVD**: Finding the normal to the centroid hyperplane is a multi-step process.
    1.  First, $N-1$ vectors that lie *within* the hyperplane are created by taking the difference between the centroids (e.g., $\mathbf{v}_i = \mathbf{c}_{i+1} - \mathbf{c}_1$ for $i=1, \dots, N-1$).
    2.  A normal vector, $\mathbf{n}_{\text{centroid}}$, must be orthogonal to all of these difference vectors.
    3.  This orthogonal vector is found by identifying the **null space** of the matrix $V$ formed by these difference vectors.
    4.  **Singular Value Decomposition (SVD)** is used on matrix $V$ to find this null space. The last right singular vector from the SVD corresponds to the null space and serves as the unit normal vector $\mathbf{n}_{\text{centroid}}$.

* **Reference Normal**: The normal vector for the reference hyperplane $\sum x_i = 0$ is straightforward.
    1.  The unscaled normal vector, $\mathbf{n}_{\text{opposite}}$, is a vector of all ones: $(1, 1, \dots, 1)$.
    2.  This vector is then normalized to have a unit length of 1.

***

### 3. Measuring Parallelism and Calculating the Final Score 🏁

The final score is based on the angle between the two normal vectors.

* **Cosine Similarity**: The parallelism between the hyperplanes is quantified by the **absolute cosine similarity** of their normal vectors. Because both $\mathbf{n}_{\text{centroid}}$ and the normalized $\mathbf{n}_{\text{opposite}}$ are unit vectors, this similarity is calculated simply as the absolute value of their dot product:

    $$
    \text{Similarity} = |\mathbf{n}_{\text{centroid}} \cdot \mathbf{n}_{\text{opposite}}|
    $$

    A value of 1 indicates the hyperplanes are perfectly parallel, while 0 means they are orthogonal.

* **Final Score**: The function returns a final score that is inversely related to the parallelism and scaled by an input factor, $f_k$.
    
    $$
    \text{Final Score} = (1 - \text{Similarity}) \cdot (1 - f_k)
    $$
