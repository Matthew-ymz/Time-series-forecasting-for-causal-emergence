import numpy as np
from scipy.linalg import sqrtm, inv
import matplotlib.pyplot as plt
from scipy.linalg import eig
from cycler import cycler
import pandas as pd
import math
import seaborn as sns
from scipy.optimize import curve_fit

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

def svd_jacs(test_id_first, start, end, interval, seed, abs_bool):
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
                
            u, s, vt = cal_W(mat, L, abs_bool)
            jacs[i].append(s)
            us[i].append(u)
            vts[i].append(vt)
            mats[i].append(mat)
            Sigs[i].append(L)

    return jacs, us, vts, mats, Sigs

def plot_singular(test_id_first, seed = 0, window='all', start=1, end=1000, interval=1, log_bool=False, abs_bool=False, leg_show=False):
    singular, us, vts, mats, Sigs = svd_jacs(test_id_first, start, end, interval, seed, abs_bool)
    num_lines = (end - start) // interval + 1
    cmap = plt.cm.viridis
    for k in range(seed):
        plt.figure(figsize=(10,8),dpi=150)
        for idx, i in enumerate(range(start, end, interval)):
            jac_arr = np.array(singular[i][k])
            if window == 'all':
                window = len(jac_arr)
            if log_bool:
                jac_arr = np.log(jac_arr)
                
            # 归一化 idx 到0-1区间，做为colormap的输入
            color_val = idx / (num_lines - 1)
            color = cmap(color_val)
            alpha = 0.5 + 0.5 * color_val

            plt.plot([j+1 for j in range(window)],jac_arr[:window], label=f'time_{i}', color=color, alpha=alpha)
        plt.title(f'seed_{k}')
        plt.xlabel('singular value index')
        plt.ylabel('singular value')
        if leg_show:
            plt.legend()
        plt.show()

    return singular, us, vts, mats, Sigs

# def analysis_u(ss, us, seq_len, dims, start, end, interval, target=[0], space_only=False, windows='all', seed=0):
#     if space_only:
#         for i in range(start, end, interval):
#             u = np.array(us[i])
#             s = np.array(ss[i])[seed, :, :]
#             u = u @ np.log(np.diag(s))
#             u_col1 = u[seed, :, :]
#             u_col1 = u_col1.reshape(seq_len,dims,-1)
#             u_col1 = np.abs(u_col1)
#             u_col1 = np.mean(u_col1, axis=0)
#             plt.figure(dpi=100)
#             if windows=='all':
#                 sns.heatmap(u_col1.T)
#             else:
#                 sns.heatmap(u_col1.T[:windows,:])
#             plt.ylabel('macro dim')
#             plt.xlabel('micro dim')
#             plt.title(str(i))
#             plt.show()
#             plt.close()
#     else:
#         for j in target:
#             for i in range(start, end, interval):
#                 u = np.array(us[i])
#                 u_col1 = u[seed, :, j]
#                 u_col1 = u_col1.reshape(seq_len,dims)
#                 u_col1 = np.abs(u_col1)
#                 plt.figure(dpi=100)
#                 sns.heatmap(u_col1.T)
#                 plt.ylabel('original dim')
#                 plt.xlabel('time')
#                 plt.title(str(i)+"_index={0}".format(j))
#                 plt.show()
#                 plt.close()
    
def analysis_u(us, dims, start, end, interval, macro_dim, seq_len=1, abs_bool=False):
    for i in range(start, end, interval):
        u = np.array(us[i])
        u_col1 = u[0, :, :macro_dim]
        if seq_len > 1:
            u_col1 = u_col1.reshape(seq_len,dims,macro_dim)
        if abs_bool:
            u_col1 = np.abs(u_col1)
        n_ticks = u_col1.shape[1]
        plt.figure(dpi=100)
        sns.heatmap(u_col1.T, xticklabels=range(1, dims + 1), yticklabels=range(1, macro_dim + 1))
        plt.ylabel('macro dim')
        plt.xlabel('micro dim')
        plt.title(f'time_{i}')
        plt.show()
            
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

def cal_W(A, Sigma, abs_bool=False):
    n = A.shape[0]
    Sigma_pinv = np.linalg.inv(Sigma) #np.linalg.pinv(Sigma, rcond=1e-15)
    matrix_a = np.conj(A).T @ Sigma_pinv @ A
    matrix_b = Sigma_pinv
    block_matrix = create_block_diagonal_matrix(matrix_a, matrix_b)
    U, S, VT = np.linalg.svd(block_matrix)
    
    U2 = U[:n,:] + U[n:,:]
    U2 = U2 @ np.diag(S)
    U2U, S2, V2T = np.linalg.svd(U2)
    if abs_bool:
        U2U = np.abs(U2U)
#     if eps != "all":
#         m = np.sum(S2 > float(eps)) 
#         if m==0:
#             m = 1
#         U2U = U2U[:, :m]
#         V2T = V2T[:, :m]
#         S2 = S2[:m]
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

def micro_analysis_jac(test_id_first, period, dims, seed, causes='all', effects='all', eps=0):
    test_id = test_id_first + str(seed)
    if causes == 'all':
        causes = range(dims)
    if effects == 'all':
        effects = range(dims)
    for cause in causes:
        for effect in effects:#[10,20,30,50,70]:
            str_i = f'{period:04d}'
            matrix = np.load(f'../results/jacobian/{test_id}/jac_{str_i}.npy')
            mat = np.abs(np.real(matrix))
            #mat = np.log(mat)
            mat_extract = extract_rows_with_interval(mat, start_row_index=cause, interval=dims)
            mat_extract = extract_cols_with_interval(mat_extract, start_col_index=effect, interval=dims)

            if mat.size > 0 and np.nanmin(mat) != np.nanmax(mat):
                min_val = np.nanmin(mat) # 使用 nanmin 来忽略NaN值
                max_val = np.nanmax(mat) # 使用 nanmax 来忽略NaN值
            elif mat.size > 0: # 如果所有值相同
                min_val = np.nanmin(mat) - 0.5 # 创建一个小的范围
                max_val = np.nanmax(mat) + 0.5
                if min_val == max_val : # 如果减去0.5后仍然相同（比如只有一个元素）
                        min_val = min_val - 1 # 确保有一个范围
                        max_val = max_val +1
            else: # 如果mat为空
                print(f"Skipping empty or invalid mat at index {i}")
                continue
            if mat_extract.max() > eps:
                sns.heatmap(mat_extract,vmin=min_val, vmax=max_val, cmap="viridis") # cmap可以自行选择
                plt.title("{0} {1} to {2}".format(period, cause, effect))
                plt.show()
                
def micro_analysis_ig(test_id_first, period, dims, seed, causes='all', effects='all', eps=0):
    test_id = test_id_first + str(seed)
    if causes == 'all':
        causes = range(dims)
    if effects == 'all':
        effects = range(dims)
    for cause in causes:
        for effect in effects:
            str_i = f'{period:04d}'
            matrix = np.load(f'../results/causal_net/{test_id}/ca_{str_i}.npy')
            mat_extract = matrix[:, cause, :, effect]

            if matrix.size > 0 and np.nanmin(matrix) != np.nanmax(matrix):
                min_val = np.nanmin(matrix) 
                max_val = np.nanmax(matrix) 
            elif matrix.size > 0: 
                min_val = np.nanmin(matrix) - 0.5 
                max_val = np.nanmax(matrix) + 0.5
            else: 
                print(f"Skipping empty or invalid mat at index {i}")
                continue
            if mat_extract.max() > eps:
                sns.heatmap(mat_extract,vmin=min_val, vmax=max_val, cmap="viridis") # cmap可以自行选择
                plt.title("{0} {1} to {2}".format(period, cause, effect))
                plt.show()
                                                                 
def dgbd_model(r, A, a, b, P):
    return A * (P + 1 - r) ** b / r ** a

# 拟合，不包括P参数，P为常数
def dgbd_model_noP(r, A, a, b):
    P = r.max()
    return dgbd_model(r, A, a, b, P)

def dgbd_fit(x, y, p0=[1, 1, 1], title='dgbd'):
    popt, pcov = curve_fit(dgbd_model_noP, x, y, p0=p0)

    print("拟合参数A, a, b:", popt)

    # 拟合曲线
    t_fit = dgbd_model_noP(x, *popt)

    plt.scatter(x, y, label='Data')
    plt.plot(x, t_fit, color='red', label='DGBD Fit')
    plt.xlabel('index')
    plt.ylabel('non-negative log value')
    plt.title(title)
    plt.legend()
    plt.show()
    
    return popt

def dgbd_ce(singular,start, end, interval, seed=1):
    A_dic = {}
    A_dic_std={}
    a_dic={}
    a_dic_std={}
    b_dic={}
    b_dic_std={}
    for i in range(start, end, interval):
        A_ls = []
        a_ls = []
        b_ls = []
        for k in range(seed):
            s_log = np.log(singular[i][k])
            s_min = np.min(s_log)
            if s_min < 0:
                s_log = s_log + np.abs(s_min)
            x = np.arange(1, len(s_log)+1) 
            param = dgbd_fit(x, s_log, p0=[0.5, 0.5, 0.5], title=str(i))
            A_ls.append(param[0])
            a_ls.append(param[1])
            b_ls.append(param[2])
        
        A_dic[i] = np.mean(A_ls)
        A_dic_std[i] = np.std(A_ls)
        a_dic[i] = np.mean(a_ls)
        a_dic_std[i] = np.std(a_ls)
        b_dic[i] = np.mean(b_ls)
        b_dic_std[i] = np.std(b_ls)
        
    plot_ce_index(A_dic, A_dic_std, y_lab = "A")
    plot_ce_index(a_dic, A_dic_std, y_lab = "a")
    plot_ce_index(b_dic, A_dic_std, y_lab = "b")
    
def plot_ce_index(val_dic, val_std, y_lab = "gini"):
    x_labels0 = list(val_dic.keys())
    x_labels = [str(i) for i in x_labels0]
    y_val = np.array(list(val_dic.values()))
    y_val_std = np.array(list(val_std.values()))
    tick_positions = np.arange(len(x_labels))
    marker_x_positions = tick_positions + 0.5

    plt.figure(figsize=(8, 4))
    plt.plot(marker_x_positions, y_val, marker='o', linestyle='-')
    plt.fill_between(marker_x_positions, y_val+y_val_std, y_val-y_val_std, alpha=0.15)
    plt.ylabel(y_lab)
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
    
# def macro_dim(us, A_mats, Sigs, start, end, interval):
#     for i in range(start, end, interval):
#         A_macro = us[i][0].T @ A_mats[i][0] @  us[i][0] 
#         Sig_macro = us[i][0].T @ Sigs[i][0] @  us[i][0] 
#         A_macro = np.real(A_macro)
#         Sig_macro = np.real(Sig_macro)
#         plt.figure(figsize=(7, 5)) # Adjust figure size as needed
#         # Using 'crest' colormap (perceptually uniform, sequential)
#         sns.heatmap(A_macro,
#                     annot=True,             # Show values in cells
#                     fmt=".2f",              # Format annotations to 2 decimal places
#                     cmap="crest",           # Uncommon colormap
#                     linewidths=.5,          # Add lines between cells
#                     linecolor='gray',       # Color of the lines
#                     cbar=True)              # Show color bar
#         plt.title(f"A_macro "+str(i), fontsize=16)
#         plt.xlabel("Dimension Index", fontsize=12)
#         plt.ylabel("Dimension Index", fontsize=12)
#         plt.xticks(rotation=45, ha='right', fontsize=10)
#         plt.yticks(rotation=0, fontsize=10)
#         plt.tight_layout() # Adjust layout to prevent labels from overlapping
#         plt.show()

#         # --- Plotting Sig_macro ---
#         plt.figure(figsize=(7, 5))
#         sns.heatmap(Sig_macro,
#                     annot=True,
#                     fmt=".2f", # Or ".2e" if scales vary widely
#                     cmap="coolwarm",       # Another diverging colormap option
#                     center=0,
#                     linewidths=.5,
#                     linecolor='gray',
#                     cbar=True) # Updated cbar label
#         plt.title(f"Sig_macro "+str(i), fontsize=16) # Updated title
#         plt.xlabel("Dimension Index", fontsize=12)
#         plt.ylabel("Dimension Index", fontsize=12)
#         plt.xticks(rotation=45, ha='right', fontsize=10)
#         plt.yticks(rotation=0, fontsize=10)
#         plt.tight_layout()
#         plt.show()