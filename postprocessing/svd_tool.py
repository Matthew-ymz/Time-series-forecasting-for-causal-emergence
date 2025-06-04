import numpy as np
from scipy.linalg import sqrtm, inv
import matplotlib.pyplot as plt
from scipy.linalg import eig
from cycler import cycler
import pandas as pd
import math
import seaborn as sns

def get_positive_contributions(jac_arr):    
    ave_sig = []
    for i in range(1, jac_arr.shape[1]+1)[::-1]:
        ave_sig.append(np.mean(jac_arr[0][0:i]))

    output = []
    for id in range(len(ave_sig)-1):
        diff = ave_sig[id+1] - ave_sig[id]
        output.append(diff)
    return output

def compute_entropy(increments):
    if not increments:
        return 0.0
    
    total = sum(increments)
    # If total is 0, there's no variation => 0.0 entropy
    if total == 0:
        return 0.0
    
    # Normalize to probabilities
    probabilities = [x / total for x in increments]

    # Compute Shannon entropy (base 2)
    entropy = 0.0
    for p in probabilities:
        # Only compute for p > 0 to avoid math domain errors
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy

def gini(jac_mean):
    return np.cumsum(jac_mean)[-1] / len(jac_mean) - 0.5

def svd_jacs(test_id_first, start, end, interval, seed, eps):
    jacs = {i:[] for i in range(start, end, interval)}
    us = {i:[] for i in range(start, end, interval)}
    vts = {i:[] for i in range(start, end, interval)}
    mats = {i:[] for i in range(start, end, interval)}
    Sigs = {i:[] for i in range(start, end, interval)}
    for k in range(seed):
        test_id = test_id_first + str(k)
        for i in range(start, end, interval):
            str_i = f'{i:04d}'
            mat = np.load(f'../results/jacobian/{test_id}/jac_{str_i}.npy')
            L = np.load(f'../results/cov_L/{test_id}/L_{str_i}.npy')
                
            u, s, vt = cal_W(mat, L, eps)
            jacs[i].append(s)
            us[i].append(u)
            vts[i].append(vt)
            mats[i].append(mat)
            Sigs[i].append(L)

    return jacs, us, vts, mats, Sigs

def plot_singular_cum(test_id_first, eps = 'all', seed = 0, window=5, window2='all', start=1, end=16435, interval=108, log_bool=False):
    singular, us, vts, mats, Sigs = svd_jacs(test_id_first, start, end, interval, seed, eps)
    gn_dic = {}
    gn_dic_std = {}
    fig1, ax1 = plt.subplots(figsize=(4, 3), dpi=100)
    fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=100)

    for i in range(start, end, interval):
        gn_ls = []
        for k in range(seed):
            jac_arr = np.array(singular[i][k])
            if log_bool:
                jac_arr = np.log(jac_arr)
            jac_mean = jac_arr #np.mean(jac_arr, axis=0)
            jac_mean_cum = np.cumsum(jac_mean)
            jac_mean_cum /= jac_mean_cum[-1]
            gn = gini(jac_mean_cum)
            gn_ls.append(gn)
        
        gn_dic[i] = np.mean(gn_ls)
        gn_dic_std[i] = np.std(gn_ls)
        # entropy_dic[time_str] = entropy
        ax1.plot(jac_mean[:window], 
                label=str(i), 
                alpha=0.5)
        # plt.fill_between(range(window), jac_mean[:window]+jac_std[:window], jac_mean[:window]-jac_std[:window], alpha=0.15)

        if window2 == 'all':
            ax2.plot(jac_mean_cum, label=str(i), alpha=0.5)
        else:
            ax2.plot(jac_mean_cum[:window2], label=str(i), alpha=0.5)
    ax2.set_xlabel("singular value index")
    #plt.xticks(range(window), [1+i for i in range(window)])
    ax2.set_ylabel("singular cumsum(normalized)")
    # plt.ylim([0.4,1])
    # plt.xlim([0,50])
    ax2.legend(loc=[1.01,0])

    ax1.set_xlabel("singular value index")
    ax1.set_xticks(range(window), [1+i for i in range(window)])
    ax1.set_ylabel("singular value")
    ax1.legend(loc=[1.01,0])
 
    return gn_dic, gn_dic_std, singular, us, vts, mats, Sigs

def analysis_u(us, seq_len, dims, start, end, interval, target):
    for j in target:
        for i in range(start, end, interval):
            u = np.array(us[i])
            u_col1 = u[0, :, j]
            u_col1 = u_col1.reshape(seq_len,dims)
            u_col1 = np.abs(u_col1)
            n_ticks = u_col1.shape[1]
            plt.figure(dpi=100)
            sns.heatmap(u_col1.T)
            plt.ylabel('original dim')
            plt.xlabel('time')
#             plt.xticks(ticks=np.arange(n_ticks) + 0.5, labels=np.arange(n_ticks), rotation=45, fontsize=6)
            plt.title(str(i)+"_index={0}".format(j))
            
def create_block_diagonal_matrix(matrix1, matrix2):
    if matrix1.ndim != 2 or matrix2.ndim != 2:
        raise ValueError("The input matrix must be two-dimensional.")
    if matrix1.shape[0] != matrix1.shape[1] or matrix2.shape[0] != matrix2.shape[1]:
        raise ValueError("The input matrix must be a square matrix.")
    n1 = matrix1.shape[0]
    n2 = matrix2.shape[0]
    result_dim = n1 + n2
    result = np.zeros((result_dim, result_dim), dtype=matrix1.dtype)
    result[:n1, :n1] = matrix1
    result[n1:, n1:] = matrix2
    return result

def cal_W(A, Sigma, eps):
    n = A.shape[0]
    Sigma_pinv = np.linalg.pinv(Sigma, rcond=1e-15)
    matrix_a = np.conj(A).T @ Sigma_pinv @ A
    matrix_b = Sigma_pinv
    block_matrix = create_block_diagonal_matrix(matrix_a, matrix_b)
    U, S, VT = np.linalg.svd(block_matrix)
    if eps == "all":
        U = np.abs(U)
    else:
        m = np.sum(S > float(eps)) 
        if m==0:
            m = 1
        U = np.abs(U)[:, :m]
        S = S[:m]
    U2 = U[:n,:] + U[n:,:]
    U2 = U2 @ np.diag(S)
    U2U, S2, V2T = np.linalg.svd(U2)
    if eps != "all":
        m = np.sum(S2 > float(eps)) 
        if m==0:
            m = 1
        U2U = U2U[:, :m]
        V2T = V2T[:, :m]
        S2 = S2[:m]
    return U2U, S2, V2T

def extract_rows_with_interval(matrix, start_row_index, interval=37):
    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.array(matrix)
        except Exception as e:
            print(f"错误：无法将输入转换为NumPy数组: {e}")
            return np.array([]) # 返回一个空数组

    # 检查矩阵是否至少是二维的
    if matrix.ndim < 2:
        print("错误：输入矩阵的维度必须至少为2。")
        return np.array([])

    num_rows = matrix.shape[0]

    # 检查起始行索引是否有效
    if start_row_index < 0 or start_row_index >= num_rows:
        print(f"错误：起始行索引 {start_row_index} 超出了矩阵的有效行范围 [0, {num_rows - 1}]。")
        return np.array([]) # 返回一个空数组

    # 生成要提取的行的索引列表
    # range(start, stop, step)
    row_indices_to_extract = list(range(start_row_index, num_rows, interval))
    print(row_indices_to_extract)

    if not row_indices_to_extract:
        print("根据给定的起始行和间隔，没有行可以被提取。")
        return np.array([])

    # 使用高级索引直接从原矩阵中提取这些行
    extracted_matrix = matrix[row_indices_to_extract, :] # ':' 表示提取所有列

    return extracted_matrix

def extract_cols_with_interval(matrix, start_col_index, interval=37):
    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.array(matrix)
        except Exception as e:
            print(f"错误：无法将输入转换为NumPy数组: {e}")
            return np.array([]) # 返回一个空数组

    # 检查矩阵是否至少是二维的
    if matrix.ndim < 2:
        print("错误：输入矩阵的维度必须至少为2。")
        return np.array([])

    num_cols = matrix.shape[1] # 获取总列数

    # 检查起始列索引是否有效
    if start_col_index < 0 or start_col_index >= num_cols:
        print(f"错误：起始列索引 {start_col_index} 超出了矩阵的有效列范围 [0, {num_cols - 1}]。")
        return np.array([]) # 返回一个空数组

    # 生成要提取的列的索引列表
    # range(start, stop, step)
    col_indices_to_extract = list(range(start_col_index, num_cols, interval))

    if not col_indices_to_extract:
        print("根据给定的起始列和间隔，没有列可以被提取。")
        return np.array([])

    # 使用高级索引直接从原矩阵中提取这些列
    # ':' 表示提取所有行
    extracted_matrix = matrix[:, col_indices_to_extract]

    return extracted_matrix

def plot_gini(gn, gn_std):
    x_labels0 = list(gn.keys())
    x_labels = [i for i in x_labels0]
    y_val = np.array(list(gn.values()))
    y_val_std = np.array(list(gn_std.values()))
    tick_positions = np.arange(len(x_labels))
    marker_x_positions = tick_positions + 0.5

    plt.figure(figsize=(8, 4))
    plt.plot(marker_x_positions, y_val, marker='o', linestyle='-')
    plt.fill_between(marker_x_positions, y_val+y_val_std, y_val-y_val_std, alpha=0.15)
    plt.ylabel("gini")
    plt.xticks(ticks=tick_positions, labels=x_labels, rotation=45, ha='right')
    if len(marker_x_positions) > 0:
        plt.xlim(tick_positions[0] - 0.5, tick_positions[-1] + 0.5 + 0.5)
    plt.tight_layout()
    plt.show()

    
def log_max_abs_eig(matrix_dic):
    """
    Compute the natural log of the absolute largest eigenvalue for each matrix
    
    Returns:
    list: Natural logs of the absolute largest eigenvalues
    """
    results = []
    x_labels = list(matrix_dic.keys())
    matrix_list = list(matrix_dic.values())
    for idx, matrix in enumerate(matrix_list):
        # Verify matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix at index {idx} is not square")
            
        # Compute eigenvalues
        eigvals = np.linalg.eigvals(matrix)
        
        # Find absolute largest eigenvalue
        abs_eigvals = np.abs(eigvals)
        max_abs_eig = np.max(abs_eigvals)
        
        # Handle edge case where eigenvalue could be zero
        if max_abs_eig == 0:
            # Zero eigenvalues need special handling for log
            results.append(-np.inf)
        else:
            results.append(np.log(max_abs_eig))
    
    return x_labels,results

def plot_eig_results(x_labels,log_values):
    """
    Create a professional line plot of the eigenvalue logs
    
    Args:
    log_values (list): Natural log values from log_max_abs_eig()
    """
    tick_positions = np.arange(len(x_labels))
    marker_x_positions = tick_positions + 0.5
    
    plt.figure(figsize=(10, 6))
    
    # Create plot with custom styling
    plt.plot(marker_x_positions,log_values, 
             marker='o', 
             markersize=8, 
             linewidth=2.5, 
             color='#1f77b4',
             markerfacecolor='white',
             markeredgewidth=2)
    
    # Customize plot aesthetics
    plt.ylabel('ln(|λ|_max)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(ticks=tick_positions, labels=x_labels, rotation=45, ha='right')
    plt.yticks(fontsize=10)
    
    # Annotate values on the plot
    for i, val in enumerate(log_values):
        plt.annotate(f'{val:.2f}', 
                     (i, val), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=9)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def macro_analysis(us, mats, start, end, interval):
    A_ma_ls = {}
    for i in range(start, end, interval):
        A_macro = us[i][0].T @ mats[i][0] @  us[i][0] 
        A_macro = np.real(A_macro)
        A_ma_ls[str(i)] = A_macro

    x_labels,results = log_max_abs_eig(A_ma_ls)
    plot_eig_results(x_labels,results)
    
def macro_dim(us, A_mats, Sigs, start, end, interval):
    for i in range(start, end, interval):
        A_macro = us[i][0].T @ A_mats[i][0] @  us[i][0] 
        Sig_macro = us[i][0].T @ Sigs[i][0] @  us[i][0] 
        A_macro = np.real(A_macro)
        Sig_macro = np.real(Sig_macro)
        plt.figure(figsize=(7, 5)) # Adjust figure size as needed
        # Using 'crest' colormap (perceptually uniform, sequential)
        sns.heatmap(A_macro,
                    annot=True,             # Show values in cells
                    fmt=".2f",              # Format annotations to 2 decimal places
                    cmap="crest",           # Uncommon colormap
                    linewidths=.5,          # Add lines between cells
                    linecolor='gray',       # Color of the lines
                    cbar=True)              # Show color bar
        plt.title(f"A_macro "+str(i), fontsize=16)
        plt.xlabel("Dimension Index", fontsize=12)
        plt.ylabel("Dimension Index", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()

        # --- Plotting Sig_macro ---
        plt.figure(figsize=(7, 5))
        sns.heatmap(Sig_macro,
                    annot=True,
                    fmt=".2f", # Or ".2e" if scales vary widely
                    cmap="coolwarm",       # Another diverging colormap option
                    center=0,
                    linewidths=.5,
                    linecolor='gray',
                    cbar=True) # Updated cbar label
        plt.title(f"Sig_macro "+str(i), fontsize=16) # Updated title
        plt.xlabel("Dimension Index", fontsize=12)
        plt.ylabel("Dimension Index", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.show()