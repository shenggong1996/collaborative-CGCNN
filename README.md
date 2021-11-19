# collaborative-CGCNN
This repo contains the codes for collabortively learning materials properties by CGCNN.

Most of the codes are revised from the CGCNN, please see the CGCNN page for reference:   https://github.com/txie-93/cgcnn.
If you see the codes here, please also cite: 10.1103/PhysRevLett.120.145301

For efficiency, we recommend doing normalization and converting graphs out of the learning loop. Please see split.py (where random spliting and robust normalization are done. Note that in this example the random spliting is designed for the task of "database completion".) and make_graphs.py for reference.
