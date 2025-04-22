# Fuzzy C-Means Clustering Project

Dự án này triển khai thuật toán Fuzzy C-Means (FCM) để phân cụm ảnh, với hai phiên bản: tuần tự và song song sử dụng MPI.

## Các file trong dự án

### 1. TuanAnh-Seq.py
File này chứa phiên bản tuần tự (sequential) của thuật toán Fuzzy C-Means.

**Tác dụng:**
- Thực hiện phân cụm FCM trên một ảnh đầu vào
- Xử lý tuần tự trên một luồng đơn
- Hiển thị kết quả phân cụm và lưu các cụm thành các ảnh riêng biệt

**Cách sử dụng:**
```bash
python TuanAnh-Seq.py
```

### 2. MPI-parrallel.py
File này chứa phiên bản song song (parallel) của thuật toán Fuzzy C-Means sử dụng MPI (Message Passing Interface).

**Tác dụng:**
- Thực hiện phân cụm FCM trên một ảnh đầu vào
- Phân phối tính toán trên nhiều CPU cores sử dụng MPI
- Tăng tốc độ xử lý bằng cách tận dụng tính toán song song
- Hiển thị kết quả phân cụm và lưu các cụm thành các ảnh riêng biệt

**Cách sử dụng:**
```bash
mpiexec -n <số_processes> python MPI-parrallel.py [đường_dẫn_ảnh]
```

Ví dụ:
```bash
mpiexec -n 4 python MPI-parrallel.py E://HPC//Brain//542x630.jpg
```

## So sánh hai phiên bản

| Tính năng | TuanAnh-Seq.py | MPI-parrallel.py |
|-----------|----------------|------------------|
| Mô hình xử lý | Tuần tự (1 luồng) | Song song (nhiều luồng) |
| Tốc độ | Chậm hơn với ảnh lớn | Nhanh hơn với ảnh lớn |
| Yêu cầu | Python, NumPy, OpenCV, Matplotlib | Python, NumPy, OpenCV, Matplotlib, mpi4py, MPI |
| Phù hợp với | Ảnh nhỏ, máy tính đơn nhân | Ảnh lớn, máy tính đa nhân |

## Cài đặt các thư viện cần thiết

```bash
pip install numpy opencv-python matplotlib
pip install mpi4py
```

Ngoài ra, bạn cần cài đặt một triển khai MPI trên hệ thống của mình:
- Windows: Microsoft MPI hoặc MPICH
- Linux: OpenMPI hoặc MPICH
- macOS: OpenMPI (qua Homebrew)

## Kết quả

Cả hai phiên bản đều tạo ra các file ảnh cho từng cụm:
- TuanAnh-Seq.py: `Cluster_1.jpg`, `Cluster_2.jpg`, ...
- MPI-parrallel.py: `Cluster_1_MPI.jpg`, `Cluster_2_MPI.jpg`, ...

## Lưu ý

- Số lượng processes trong MPI-parrallel.py nên phù hợp với số lượng CPU cores có sẵn trên máy tính để đạt hiệu suất tối ưu.
- Với ảnh lớn, phiên bản MPI-parrallel.py sẽ cho thấy sự cải thiện đáng kể về thời gian thực thi so với phiên bản tuần tự.
