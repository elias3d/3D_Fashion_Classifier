import streamlit as st

st.title("How it works")

st.markdown("""
This project is the 3D version of the MNIST digit classifier. Instead of a 28 × 28 drawing, the input is
a 3D garment from Maya, and instead of 10 digits the model picks one of 8 fashion types.
""")

st.markdown("""
```mermaid
flowchart LR
    A[Maya: build or import garment] --> B[Tag fashionType attribute]
    B --> C[Export FBX + triangles .npz]
    C --> D[Train VoxelCNN in PyTorch]
    D --> E[models/voxel_cnn.pt]
    E --> F[Inference server]
    F <--> G[Maya panel]
    E --> H[Streamlit app]
    H <-->|command port 7001| G
```
""")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("""
**MNIST (2D)**

1. Drawing on a canvas
2. Convert to grayscale
3. Resize to **28 × 28** pixels
4. Divide by 255 → values **0 to 1**
5. 2D CNN (`Conv2d`)
6. Softmax over **10 digits**
""")
with c2:
    with st.container(border=True):
        st.markdown("""
**Fashion3D (3D)**

1. Mesh from Maya (or FBX / OBJ upload)
2. Sample 6,000 points on the surface
3. Center + scale into [-1, 1], then voxelize to **32 × 32 × 32**
4. Log point density ÷ max → values **0 to 1**
5. 3D CNN (`Conv3d`)
6. Softmax over **8 fashion types**
""")

st.subheader("Why voxels?")
st.markdown("""
There are four common ways an AI can "look" at a 3D asset:

| Approach | What the model sees | Example models |
|---|---|---|
| Multi-view images | several 2D renders | MVCNN |
| **Voxels (used here)** | a 3D grid of filled / empty cells | VoxNet, 3D ShapeNets |
| Point clouds | a list of xyz points | PointNet, DGCNN |
| Mesh graphs | vertices + edges + faces | MeshCNN |

Voxels are the closest thing to pixels, so everything from the MNIST project carries over: a fixed-size grid,
values between 0 and 1, and convolution filters, now sliding in three directions instead of two.
No rendering is involved: the model reads the geometry directly.
""")

st.subheader("The network")
st.markdown("""
| Layer | Output shape | What it learns |
|---|---|---|
| Input | 1 × 32 × 32 × 32 | the voxel grid |
| Conv3d 5×5×5, stride 2 + BatchNorm + ReLU | 32 × 16 × 16 × 16 | surfaces and edges |
| Conv3d 3×3×3 + BatchNorm + ReLU, MaxPool | 64 × 8 × 8 × 8 | tubes, rings, flat panels |
| Conv3d 3×3×3 + BatchNorm + ReLU, MaxPool | 128 × 4 × 4 × 4 | parts: sleeves, legs, collars, soles |
| Dense 256 + Dropout | 256 | combinations of parts |
| Dense 8 + Softmax | 8 | probability of each fashion type |

**Training tricks:** each epoch every garment is randomly rotated around Y, tilted a little, and stretched
0.85–1.15× per axis (data augmentation), so the network learns *shapes*, not one fixed pose.
""")
