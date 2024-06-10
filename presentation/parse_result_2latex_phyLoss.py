import json

placeholder_value = "N/A"

def generate_na_row(model, scale):
    return f"{model} & {scale} & Bicubic & 0 & " + " & ".join([placeholder_value] * 5) + " \\\\"

def generate_row(model, data_16k, data_32k, scale):
    row = f"{model} & {scale} & Bicubic & 0"
    for data in [data_16k, data_32k]:
        physics_test_error_1 = data["metrics"]["Physics"].get("test1 error", placeholder_value)
        physics_test_error_2 = data["metrics"]["Physics"].get("test2 error", placeholder_value)
        row += f" & {physics_test_error_1:.2f}" if isinstance(physics_test_error_1, (float, int)) else f" & {placeholder_value}"
        row += f" & {physics_test_error_2:.2f}" if isinstance(physics_test_error_2, (float, int)) else f" & {placeholder_value}"
    row += f" & {data_16k['parameters']:.2f} M \\\\"
    return row

def generate_row_nosie(model, data_16k, data_32k, noise):
    row = f"{model} & 8 & Noisy Uniform & {noise}"
    for data in [data_16k, data_32k]:
        physics_test_error_1 = data["metrics"]["Physics"].get("test1 error", placeholder_value)
        physics_test_error_2 = data["metrics"]["Physics"].get("test2 error", placeholder_value)
        row += f" & {physics_test_error_1:.2f}" if isinstance(physics_test_error_1, (float, int)) else f" & {placeholder_value}"
        row += f" & {physics_test_error_2:.2f}" if isinstance(physics_test_error_2, (float, int)) else f" & {placeholder_value}"
    row += f" & {data_16k['parameters']:.2f} M \\\\"
    return row

def generate_row_lrsim(model, data_16k, data_32k, noise):
    row = f"{model} & 4 & LR Simulation & 0"
    for data in [data_16k, data_32k]:
        physics_test_error_1 = data["metrics"]["Physics"].get("test1 error", placeholder_value)
        physics_test_error_2 = data["metrics"]["Physics"].get("test2 error", placeholder_value)
        row += f" & {physics_test_error_1:.2f}" if isinstance(physics_test_error_1, (float, int)) else f" & {placeholder_value}"
        row += f" & {physics_test_error_2:.2f}" if isinstance(physics_test_error_2, (float, int)) else f" & {placeholder_value}"
    row += f" & {data_16k['parameters']:.2f} M \\\\"
    return row

def generate_latex_table_from_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    latex_table = "\\begin{table}[h!]\n"
    latex_table += "\\caption{Results of physics errors on fluid flow data.}\n"
    latex_table += "\\label{tab:phy_loss}\n"
    latex_table += "\\centering\n"
    latex_table += "\\scalebox{0.8}{\n"
    latex_table += "\\begin{tabular}{l|c|c|c|cc|cc|c}\n"
    latex_table += "\\toprule\n"
    latex_table += " & UF & Down-sampling & Noise & \\multicolumn{2}{c}{{$Re=16000$}} & \\multicolumn{2}{c}{{$Re=32000$}} & \\\\ \n"
    latex_table += "\\cmidrule(r){2-6} \\cmidrule(r){7-8}\n"
    latex_table += "Methods & ($\\times$) & & (\\%) & Interp. & Extrap. & Interp. & Extrap. & \\# par.\\\\\n"
    latex_table += "\\midrule\n"

    model_order = ["Bicubic", "SRCNN","FNO2D", "subpixelCNN", "EDSR", "WDSR", "SwinIR","SwinIR_p001"]

    scales = [8, 16]
    for scale in scales:
        for model in model_order:
            key_16k = f"{model}_nskt_16k_bicubic_{scale}_0.0"
            key_32k = f"{model}_nskt_32k_bicubic_{scale}_0.0"
            if key_16k in data and key_32k in data:
                latex_table += generate_row(model, data[key_16k], data[key_32k], scale) + "\n"
            else:
                latex_table += generate_na_row(model, scale) + "\n"
    scales = [4]
    for scale in scales:
        for model in model_order:
            key_16k = f"{model}_nskt_16k_sim_4_v8_bicubic_{scale}_0.0" #nskt_32k_sim_4_v8
            key_32k = f"{model}_nskt_32k_sim_4_v8_bicubic_{scale}_0.0"
            if key_16k in data and key_32k in data:
                print(model)
                latex_table += generate_row_lrsim(model, data[key_16k], data[key_32k], scale) + "\n"
            else:
                latex_table += generate_na_row(model, scale) + "\n"
    for noise in [0.05,0.1]:
        for model in ["SwinIR"]:
            key_16k = f"{model}_nskt_32k_noisy_uniform_8_{noise}"
            key_32k = f"{model}_nskt_32k_noisy_uniform_8_{noise}"
            if key_16k in data and key_32k in data:
                latex_table += generate_row_nosie(model, data[key_16k], data[key_32k], noise) + "\n"
            else:
                latex_table += generate_na_row(model, scale) + "\n"
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}}\n"
    latex_table += "\\end{table}\n"
    return latex_table

# Example usage
latex_table = generate_latex_table_from_json("normed_eval.json")
print(latex_table)

