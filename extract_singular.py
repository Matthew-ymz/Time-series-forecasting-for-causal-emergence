import numpy as np

if __name__ == '__main__':
    path = 'results/jacobian/long_term_forecast_ca2pABCc73dlp1_3_12_12_iTransformer_Ca2p_ftM_sl12_pl12_dm32_nh8_el4_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0/'
    file_names = [f'jac_{i:04}.npy' for i in range(0, 15000, 250)]
    singular_values = []
    for f in file_names:
        print(f)
        jac = np.load(path + f)
        jac_f = jac.reshape(12192, 12192).astype(float)
        s = np.sqrt(np.linalg.eigvalsh(jac_f @ jac_f.T)[::-1])
        singular_values.append(s)

    singular_array = np.array(singular_values)
    np.save(path + 'singular.npy', singular_array)
