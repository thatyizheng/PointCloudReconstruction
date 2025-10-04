Point Cloud Reconstruction
1. Computing disparity with RAFT-Stereo or PSMNet
2. Generating 3D point cloud with disparity
3. Projecting 3D points back to ORIGINAL left/right images (visualize uL/uR)
4. Running on image pairs OR dual live video streams.

Clean utilities in ``utils.py`` + a simple runner to visualize ``visualization.ipynb``
