building and running

1. cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
2. cmake --build build --target federated_kmeans --config Release


output:

cat kmeans-143549.out
[rank 4] loaded 1000 samples, D=3072
[rank 3] loaded 2000 samples, D=3072
[rank 2] loaded 3000 samples, D=3072
[rank 1] loaded 5000 samples, D=3072
[round 0] done
[round 0] rank 2@mscluster46.ms.wits.ac.za done
[round 0] rank 1@mscluster45.ms.wits.ac.za done
[round 0] rank 3@mscluster47.ms.wits.ac.za done
[round 0] rank 4@mscluster48.ms.wits.ac.za done
[round 1] done
[round 1] rank 2@mscluster46.ms.wits.ac.za done
[round 1] rank 3@mscluster47.ms.wits.ac.za done
[round 1] rank 4@mscluster48.ms.wits.ac.za done
[round 1] rank 1@mscluster45.ms.wits.ac.za done
[round 2] done
[round 2] rank 2@mscluster46.ms.wits.ac.za done
[round 2] rank 4@mscluster48.ms.wits.ac.za done
[round 2] rank 3@mscluster47.ms.wits.ac.za done
[round 2] rank 1@mscluster45.ms.wits.ac.za done
[round 3] done
[round 3] rank 1@mscluster45.ms.wits.ac.za done
[round 3] rank 3@mscluster47.ms.wits.ac.za done
[round 3] rank 2@mscluster46.ms.wits.ac.za done
[round 3] rank 4@mscluster48.ms.wits.ac.za done
[round 4] done
[round 4] rank 1@mscluster45.ms.wits.ac.za done
[round 4] rank 4@mscluster48.ms.wits.ac.za done
[round 4] rank 3@mscluster47.ms.wits.ac.za done
[round 4] rank 2@mscluster46.ms.wits.ac.za done
[round 5] done
[round 5] rank 1@mscluster45.ms.wits.ac.za done
[round 5] rank 2@mscluster46.ms.wits.ac.za done
[round 5] rank 3@mscluster47.ms.wits.ac.za done
[round 5] rank 4@mscluster48.ms.wits.ac.za done
[round 6] done
[round 6] rank 2@mscluster46.ms.wits.ac.za done
[round 6] rank 3@mscluster47.ms.wits.ac.za done
[round 6] rank 4@mscluster48.ms.wits.ac.za done
[round 6] rank 1@mscluster45.ms.wits.ac.za done
[round 7] done
[round 7] rank 1@mscluster45.ms.wits.ac.za done
[round 7] rank 4@mscluster48.ms.wits.ac.za done
[round 7] rank 3@mscluster47.ms.wits.ac.za done
[round 7] rank 2@mscluster46.ms.wits.ac.za done
[round 8] done
[round 8] rank 2@mscluster46.ms.wits.ac.za done
[round 8] rank 1@mscluster45.ms.wits.ac.za done
[round 8] rank 3@mscluster47.ms.wits.ac.za done
[round 8] rank 4@mscluster48.ms.wits.ac.za done
[round 9] done
[round 9] rank 1@mscluster45.ms.wits.ac.za done
[round 9] rank 4@mscluster48.ms.wits.ac.za done
[round 9] rank 2@mscluster46.ms.wits.ac.za done
[round 9] rank 3@mscluster47.ms.wits.ac.za done
[round 10] done
[round 10] rank 1@mscluster45.ms.wits.ac.za done
[round 10] rank 4@mscluster48.ms.wits.ac.za done
[round 10] rank 2@mscluster46.ms.wits.ac.za done
[round 10] rank 3@mscluster47.ms.wits.ac.za done
[round 11] done
[round 11] rank 1@mscluster45.ms.wits.ac.za done
[round 11] rank 4@mscluster48.ms.wits.ac.za done
[round 11] rank 3@mscluster47.ms.wits.ac.za done
[round 11] rank 2@mscluster46.ms.wits.ac.za done
[round 12] done
[round 12] rank 2@mscluster46.ms.wits.ac.za done
[round 12] rank 3@mscluster47.ms.wits.ac.za done
[round 12] rank 4@mscluster48.ms.wits.ac.za done
[round 12] rank 1@mscluster45.ms.wits.ac.za done
[round 13] done
[round 13] rank 1@mscluster45.ms.wits.ac.za done
[round 13] rank 4@mscluster48.ms.wits.ac.za done
[round 13] rank 3@mscluster47.ms.wits.ac.za done
[round 13] rank 2@mscluster46.ms.wits.ac.za done
[round 14] done
[round 14] rank 1@mscluster45.ms.wits.ac.za done
[round 14] rank 3@mscluster47.ms.wits.ac.za done
[round 14] rank 4@mscluster48.ms.wits.ac.za done
[round 14] rank 2@mscluster46.ms.wits.ac.za done
[round 15] done
[round 15] rank 2@mscluster46.ms.wits.ac.za done
[round 15] rank 1@mscluster45.ms.wits.ac.za done
[round 15] rank 4@mscluster48.ms.wits.ac.za done
[round 15] rank 3@mscluster47.ms.wits.ac.za done
[round 16] done
[round 16] rank 2@mscluster46.ms.wits.ac.za done
[round 16] rank 1@mscluster45.ms.wits.ac.za done
[round 16] rank 3@mscluster47.ms.wits.ac.za done
[round 16] rank 4@mscluster48.ms.wits.ac.za done
[round 17] done
[round 17] rank 1@mscluster45.ms.wits.ac.za done
[round 17] rank 2@mscluster46.ms.wits.ac.za done
[round 17] rank 3@mscluster47.ms.wits.ac.za done
[round 17] rank 4@mscluster48.ms.wits.ac.za done
[round 18] done
[round 18] rank 1@mscluster45.ms.wits.ac.za done
[round 18] rank 4@mscluster48.ms.wits.ac.za done
[round 18] rank 3@mscluster47.ms.wits.ac.za done
[round 18] rank 2@mscluster46.ms.wits.ac.za done
[round 19] done
[round 19] rank 2@mscluster46.ms.wits.ac.za done
[round 19] rank 4@mscluster48.ms.wits.ac.za done
[round 19] rank 1@mscluster45.ms.wits.ac.za done
[round 19] rank 3@mscluster47.ms.wits.ac.za done
Centroids saved to centroids.bin
Centroid[0][0..4]: 0.107843 0.156863 0.0509804 0.109804 0.156863
=== Federated K-Means Evaluation ===
K        : 10
D        : 3072
Samples  : 2100
Inertia  : 963171
Avg dist : 21.4162
Counts per cluster:
C0: 117
C1: 0
C2: 4
C3: 1979
C4: 0
C5: 0
C6: 0
C7: 0
C8: 0
C9: 0
