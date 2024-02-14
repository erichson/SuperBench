# Modifying the script to ensure that even if a model's data is not present in the JSON, 
# it still appears in the table with all its metrics set to "N/A".

placeholder_value = "INP"
# Helper function to generate a row for the LaTeX table with all values set to "N/A"
def generate_na_row(model,scale):
    row = model
    row += " & " + str(scale)
    row += " & " + " & ".join([placeholder_value] * 12)  # 12 metrics (6 interpolation + 6 extrapolation)
    row += " & " + placeholder_value  # for the parameters column
    row += " \\\\"
    return row
    # Helper function to generate a row for the LaTeX table
    
def generate_row(model, data,scale):
    row = model
    if scale in data:
        row += " & " + str(scale)
        for metric in ["MSE", "MAE", "RFNE", "IN", "PSNR", "SSIM", "MSE_2", "MAE_2", "RFNE_2", "IN_2", "PSNR_2", "SSIM_2"]:
            value = data[scale][metric]
            row += " & " + "{:.2f}".format(value) if isinstance(value, (float, int)) else " & " + value
        row += " & " + "{:.2f} M".format(data[scale]["parameters"]) + " \\\\"

    else:
        row += " & " + " ".join([placeholder_value] * 15)
        row += " \\\\"
        
    return row
    
def generate_latex_table_from_json(json_file, target_dataset):
    # Load the JSON data
    # As we don't have the file, this part is commented out
    import json
    with open(json_file, "r") as file:
        data = json.load(file)
    
    # For this demonstration, we'll use a dummy data structure
    # Extracting relevant entries
    relevant_data = {
        key: value for key, value in data.items() 
        if value["dataset"] == target_dataset and value["method"] == "bicubic" 
    }
    # Organizing the data for the table
    table_data = {}
    for key, value in relevant_data.items():
        model = value["model"]
        scale_factor = value["scale factor"]
        if model not in table_data:
            table_data[model] = {}
        table_data[model][scale_factor] = {
            metric: value["metrics"].get(metric, {}).get("test1 error", placeholder_value)
            for metric in ["MSE", "MAE", "RFNE", "IN", "PSNR", "SSIM"]
        }
        table_data[model][scale_factor].update({
            metric + "_2": value["metrics"].get(metric, {}).get("test2 error", placeholder_value)
            for metric in ["MSE", "MAE", "RFNE", "IN", "PSNR", "SSIM"]
        })
        table_data[model][scale_factor]["parameters"] = value["parameters"]

    # Generate the LaTeX table
    latex_table = "\\begin{table}[h!]\n"
    latex_table += "\\caption{Results for " + target_dataset.replace("_", "\\_") + " dataset with bicubic down-sampling.}\n"
    latex_table += "\\label{tab:" + target_dataset + "_bicubic}\n"
    latex_table += "\\centering\n"
    latex_table += "\\scalebox{0.6}{\n"
    latex_table += "\\begin{tabular}{l|c|cccccc|cccccc|c}\n"
    latex_table += "\\toprule\n"
    latex_table += "& UF & \\multicolumn{6}{c}{{Interpolation Errors}} & \\multicolumn{6}{c}{{Extrapolation Errors}} \\\\ \n"
    latex_table += "\\cmidrule(r){2-8} \\cmidrule(r){8-14} \n"
    latex_table += "Baselines & ($\\times$)& MSE& MAE & RFNE  & IN & PSNR & SSIM & MSE & MAE &RFNE & IN & PSNR  & SSIM & \\# par.\\\\ \n"
    latex_table += "\\midrule\n"

    # Add rows for each model
    model_order = ["Bicubic","FNO2D","FNO2D_patch","SRCNN", "subpixelCNN", "EDSR", "WDSR", "SwinIR"]
    for scale in [8]:
        for model in model_order:
            if (model in table_data) and (scale in table_data[model]):
                latex_table += generate_row(model, table_data[model],scale) + "\n"
            else:
                latex_table += generate_na_row(model,scale) + "\n"  # Add the model row with all N/A values

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}}\n"
    latex_table += "\\end{table}\n"
    return latex_table

# Return the modified function for review
with open("latex_table.sh", "w") as file:
    print(generate_latex_table_from_json("normed_eval.json", "cosmo_sim_8"),file=file)
