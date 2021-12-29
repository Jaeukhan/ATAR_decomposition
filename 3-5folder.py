import os

fold = ["12su","12sb", "12kh", "12dh"]
for j in range(len(fold)):
    if not os.path.exists("concat/%s"%fold[j]):
        os.mkdir("concat/%s"%fold[j])
    for i in range(5):
        os.mkdir(f"concat/{fold[j]}/{i+1}-10")
        os.mkdir(f"concat/{fold[j]}/{i + 1}-11")
        os.mkdir(f"concat/{fold[j]}/{i + 1}-15")
