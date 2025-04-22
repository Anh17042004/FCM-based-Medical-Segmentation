import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

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

    def fit(self, img_vector):
        n_points = img_vector.shape[0]
        
        # Initialize membership matrix
        level_members = np.random.rand(self.n_clusters, n_points)
        level_members = level_members/np.sum(level_members, axis=0)
        
        centers = self.compute_centers(level_members, img_vector)

        for _ in range(self.max_iter):
            new_level_members = self.update_membership(img_vector, centers)
            new_centers = self.compute_centers(new_level_members, img_vector)

            if np.linalg.norm(new_centers - centers) < self.epsilon:
                break
                
            centers = new_centers
            level_members = new_level_members

        return centers, level_members

    def visualize_results(self, original_img, clusters):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.title('Ảnh Gốc')
        plt.imshow(original_img, cmap='gray')
        plt.axis('off')

        plt.subplot(132)
        plt.title('Phân Đoạn')
        plt.imshow(clusters, cmap='viridis')
        plt.axis('off')

        # Hiển thị các cụm riêng biệt
        plt.subplot(133)
        plt.title('Cụm Đầu Tiên')
        cluster_img = np.zeros_like(original_img)
        cluster_img[clusters == 0] = original_img[clusters == 0]
        plt.imshow(cluster_img, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def main():
    # Đọc ảnh
    img_path = "nao.jpg"
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Không thể đọc ảnh từ {img_path}")
        return

    # Chuẩn bị dữ liệu
    img_vector = img.flatten()
    
    # Khởi tạo và chạy FCM
    fcm = FuzzyCMeans(n_clusters=5)
    
    start_time = time.time()
    centers, level_members = fcm.fit(img_vector)
    clusters = np.argmax(level_members, axis=0).reshape(img.shape)
    end_time = time.time()
    
    print(f"Thời gian thực thi: {end_time - start_time:.5f} giây")
    
    # Hiển thị kết quả
    fcm.visualize_results(img, clusters)
    
    # Lưu các cụm thành ảnh riêng
    for cluster_idx in range(fcm.n_clusters):
        cluster_img = np.zeros_like(img)
        cluster_img[clusters == cluster_idx] = img[clusters == cluster_idx]
        cv.imwrite(f"Cluster_{cluster_idx + 1}.jpg", cluster_img)

if __name__ == "__main__":
    main()
