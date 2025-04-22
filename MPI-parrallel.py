"""MPI-Parallel.py - Fuzzy C-Means Clustering with MPI Parallelization

Chương trình này thực hiện thuật toán Fuzzy C-Means (FCM) song song sử dụng MPI
để phân phối tính toán trên nhiều CPU cores, giúp tăng tốc độ xử lý.

Cách sử dụng:
1. Cài đặt thư viện mpi4py: pip install mpi4py
2. Cài đặt MPI trên hệ thống (MPICH hoặc OpenMPI)
3. Chạy chương trình với lệnh: mpiexec -n <số_processes> python MPI-parrallel.py
   Ví dụ: mpiexec -n 4 python MPI-parrallel.py

Lưu ý: Số processes nên phù hợp với số lượng CPU cores có sẵn trên máy tính.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("Thư viện mpi4py chưa được cài đặt. Hãy cài đặt bằng lệnh: pip install mpi4py")
    sys.exit(1)

class FuzzyCMeans:
    def __init__(self, n_clusters=5, m=2, max_iter=100, epsilon=1e-8):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.epsilon = epsilon

    def compute_centers(self, level_members, img_vector):
        centers = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            numerator = np.sum((level_members[i]**self.m)*img_vector)
            denominator = np.sum(level_members[i]**self.m)
            centers[i] = numerator/denominator
        return centers

    def update_membership(self, img_vector, centers):
        n_points = img_vector.shape[0]
        level_members = np.zeros((self.n_clusters, n_points))
        distances = np.transpose(abs(img_vector[:, np.newaxis] - centers))
        for i in range(self.n_clusters):
            level_members[i] = 1/np.sum((distances[i]/distances)**(2/(self.m-1)), axis=0)
        return level_members

    def fit_parallel(self, img_vector, comm, rank, size):
        n_points = img_vector.shape[0]

        # Distribute data across processes
        local_size = n_points // size
        remainder = n_points % size

        # Calculate local data sizes for each process
        counts = [local_size + 1 if i < remainder else local_size for i in range(size)]
        displacements = [sum(counts[:i]) for i in range(size)]

        # Allocate local data for this process
        local_size = counts[rank]
        local_img_vector = np.zeros(local_size, dtype=np.float64)

        # Scatter the image data
        comm.Scatterv([img_vector, counts, displacements, MPI.DOUBLE], local_img_vector)

        # Initialize membership matrix (only on rank 0)
        if rank == 0:
            level_members = np.random.rand(self.n_clusters, n_points)
            level_members = level_members/np.sum(level_members, axis=0)
        else:
            level_members = None

        # Broadcast initial centers from rank 0 to all processes
        if rank == 0:
            centers = self.compute_centers(level_members, img_vector)
        else:
            centers = np.zeros(self.n_clusters)

        centers = comm.bcast(centers, root=0)

        # Allocate local membership matrix
        local_level_members = np.zeros((self.n_clusters, local_size))

        for _ in range(self.max_iter):
            # Update local membership matrix
            local_level_members = self.update_membership(local_img_vector, centers)

            # Gather partial numerators and denominators for center calculation
            local_numerators = np.zeros(self.n_clusters)
            local_denominators = np.zeros(self.n_clusters)

            for i in range(self.n_clusters):
                local_numerators[i] = np.sum((local_level_members[i]**self.m)*local_img_vector)
                local_denominators[i] = np.sum(local_level_members[i]**self.m)

            # Reduce to get global numerators and denominators
            global_numerators = np.zeros(self.n_clusters)
            global_denominators = np.zeros(self.n_clusters)

            comm.Allreduce(local_numerators, global_numerators, op=MPI.SUM)
            comm.Allreduce(local_denominators, global_denominators, op=MPI.SUM)

            # Calculate new centers
            new_centers = global_numerators / global_denominators

            # Check convergence
            if np.linalg.norm(new_centers - centers) < self.epsilon:
                break

            centers = new_centers

        # Gather all local membership matrices to rank 0
        if rank == 0:
            level_members = np.zeros((self.n_clusters, n_points))

        # Gather the local membership matrices
        for i in range(self.n_clusters):
            recvbuf = None
            if rank == 0:
                recvbuf = np.zeros(n_points)

            comm.Gatherv(local_level_members[i], [recvbuf, counts, displacements, MPI.DOUBLE], root=0)

            if rank == 0:
                level_members[i] = recvbuf

        return centers, level_members

    def visualize_results(self, original_img, clusters):
        # Hiển thị ảnh gốc và phân đoạn
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.title('Ảnh Gốc')
        plt.imshow(original_img, cmap='gray')
        plt.axis('off')

        plt.subplot(122)
        plt.title('Phân Đoạn')
        plt.imshow(clusters, cmap='viridis')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Hiển thị từng cụm riêng biệt
        n_rows = (self.n_clusters + 2) // 3  # Số hàng cần thiết để hiển thị tất cả các cụm
        plt.figure(figsize=(15, 5 * n_rows))

        for cluster_idx in range(self.n_clusters):
            plt.subplot(n_rows, 3, cluster_idx + 1)
            plt.title(f'Cụm {cluster_idx + 1}')
            cluster_img = np.zeros_like(original_img)
            cluster_img[clusters == cluster_idx] = original_img[clusters == cluster_idx]
            plt.imshow(cluster_img, cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only rank 0 reads the image
    if rank == 0:
        # Xử lý tham số dòng lệnh để lấy đường dẫn ảnh
        default_img_path = "nao.jpg"

        if len(sys.argv) > 1:
            img_path = sys.argv[1]
        else:
            img_path = default_img_path
            print(f"Khong co duong dan anh duoc cung cap. Su dung duong dan mac dinh: {img_path}")

        # Đọc ảnh
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Không thể đọc ảnh từ {img_path}")
            print("Hãy chạy lại với đường dẫn ảnh hợp lệ: mpiexec -n <số_processes> python MPI-parrallel.py <đường_dẫn_ảnh>")
            comm.Abort()
            return

        img_shape = img.shape
        img_vector = img.flatten().astype(np.float64)
    else:
        img = None
        img_shape = None
        img_vector = None

    # Broadcast image shape to all processes
    img_shape = comm.bcast(img_shape, root=0)

    # Create a buffer for the image vector on non-root processes
    if rank != 0:
        img_vector = np.zeros(img_shape[0] * img_shape[1], dtype=np.float64)

    # Broadcast the image vector to all processes
    comm.Bcast(img_vector, root=0)

    # Khởi tạo và chạy FCM
    fcm = FuzzyCMeans(n_clusters=5)

    # Measure execution time
    if rank == 0:
        start_time = time.time()

    _, level_members = fcm.fit_parallel(img_vector, comm, rank, size)  # Sử dụng _ để bỏ qua biến centers không sử dụng

    # Only rank 0 processes the results
    if rank == 0:
        end_time = time.time()
        print(f"\n=== Thuc thi FCM song song voi {size} processes ===")
        print(f"Thoi gian thuc thi: {end_time - start_time:.5f} giay")

        # Determine cluster assignments
        clusters = np.argmax(level_members, axis=0).reshape(img_shape)

        # Hiển thị kết quả
        print("\nHien thi ket qua phan cum...")
        fcm.visualize_results(img, clusters)

        # Lưu các cụm thành ảnh riêng
        print("\nLuu cac cum thanh anh rieng:")
        for cluster_idx in range(fcm.n_clusters):
            cluster_img = np.zeros_like(img)
            cluster_img[clusters == cluster_idx] = img[clusters == cluster_idx]
            output_filename = f"Cluster_{cluster_idx + 1}_MPI.jpg"
            cv.imwrite(output_filename, cluster_img)
            print(f"  - Da luu cum {cluster_idx + 1} vao file: {output_filename}")

        print("\nHoan tat phan cum FCM song song!")

if __name__ == "__main__":
    main()


